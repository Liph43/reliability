#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生版：读取 components.csv + mission_profile.csv + model.json
- 由 mission_profile 计算每个元件 duty
- 由 model.json 计算系统任务可靠度
- 生成 output/lab1_report_<student_id>_<name>.md
- 内置 sanity checks

用法：
python src/calc.py --student_id 2026XXXXXX --student_name zhangsan --N 60
"""

import csv
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
COMP_PATH = DATA_DIR / "components.csv"
PROFILE_PATH = DATA_DIR / "mission_profile.csv"
MODEL_PATH = DATA_DIR / "model.json"

EXPERIMENT_NAME = "实验：完整搬运循环任务可靠度评估（学生版：RBD+任务剖面）"


def R_exp(lmbda: float, t: float) -> float:
    return math.exp(-lmbda * t)


def R_parallel(Rs: List[float]) -> float:
    p_fail = 1.0
    for r in Rs:
        p_fail *= (1.0 - r)
    return 1.0 - p_fail


def R_series(Rs: List[float]) -> float:
    r = 1.0
    for x in Rs:
        r *= x
    return r


def load_components() -> Dict[str, Tuple[str, float]]:
    comps = {}
    with COMP_PATH.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            cid = row["id"].strip()
            name = row["name"].strip()
            lam = float(row["lambda_per_h"])
            comps[cid] = (name, lam)
    return comps


def load_profile(component_ids: List[str]) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    """
    returns:
      t_cyc: 单循环时长（h）=五阶段时长之和
      duty_map: 每个元件 duty
      rows: profile 原始行（便于报告展示）
    """
    rows = []
    # sum working time per component
    work_time = {cid: 0.0 for cid in component_ids}
    t_cyc = 0.0

    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        missing = [c for c in ["phase", "duration_h"] + component_ids if c not in rd.fieldnames]
        if missing:
            raise ValueError(f"mission_profile.csv 缺少列：{missing}")
        for row in rd:
            phase = row["phase"].strip()
            dur = float(row["duration_h"])
            if dur <= 0:
                raise ValueError(f"阶段 {phase} 的 duration_h 必须>0")
            t_cyc += dur
            for cid in component_ids:
                flag = int(float(row[cid]))
                if flag not in (0, 1):
                    raise ValueError(f"{phase} 阶段 {cid} 标记必须为0/1")
                if flag == 1:
                    work_time[cid] += dur
            rows.append({"phase": phase, "duration_h": dur, **{cid: int(float(row[cid])) for cid in component_ids}})

    if t_cyc <= 0:
        raise ValueError("t_cyc 计算得到 <=0，请检查 mission_profile.csv")

    duty = {cid: work_time[cid] / t_cyc for cid in component_ids}
    return t_cyc, duty, rows


Node = Union[str, Dict[str, Any]]


def parse_model() -> Node:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model = data.get("model")
    if model == "__FILL__" or model is None:
        raise ValueError("data/model.json 仍为占位符 __FILL__，请先补全你的 RBD 结构。")
    return model


def eval_node(node: Node, R_map: Dict[str, float]) -> float:
    """递归计算 RBD 节点可靠度"""
    if isinstance(node, str):
        if node not in R_map:
            raise KeyError(f"模型中引用了未知元件：{node}")
        return R_map[node]
    if isinstance(node, dict):
        if "series" in node:
            return R_series([eval_node(x, R_map) for x in node["series"]])
        if "parallel" in node:
            return R_parallel([eval_node(x, R_map) for x in node["parallel"]])
        raise ValueError(f"不支持的节点类型：{node.keys()}（仅支持 series/parallel）")
    raise TypeError(f"非法节点：{type(node)}")


def strip_parallel(node: Node) -> Node:
    """用于sanity check：把并联节点退化为单支路（取第一个），其余保持"""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if "series" in node:
            return {"series": [strip_parallel(x) for x in node["series"]]}
        if "parallel" in node:
            first = node["parallel"][0]
            return strip_parallel(first)
    return node


def calculate_subsystem_reliability(R_map: Dict[str, float]) -> Dict[str, float]:
    """计算并返回各个子系统的独立可靠度"""
    subsystems = {
        "1）共用供电及控制链": {"series": ["C1", "C2", {"parallel": ["C3", "C4"]}, "C5", "C6", "C7"]},
        "2）起升执行链": {"series": ["C8", "C9", "C10", "C11"]},
        "3）小车执行链": {"series": ["C12", "C13", "C14"]},
        "4）冗余测量与安全": {"series": [{"parallel": ["C15", "C16"]}, {"parallel": ["C17", "C18"]}]},
        "5）夹具": "C19"
    }
    # 复用 eval_node 递归求解结构概率
    return {name: eval_node(struct, R_map) for name, struct in subsystems.items()}


def main():
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_id", required=True)
    ap.add_argument("--student_name", required=True)
    ap.add_argument("--N", type=int, default=60, help="班次循环次数")
    args = ap.parse_args()
    if args.N <= 0:
        raise ValueError("N 必须>0")
    # 统一命名规范
    safe_name = re.sub(r'[^\w\u4e00-\u9fa5]', '', str(args.student_name))
    OUT_PATH = REPO_ROOT / "output" / f"experiment_report_{args.student_id}_{safe_name}.md"

    comps = load_components()
    component_ids = sorted(comps.keys(), key=lambda x: int(x[1:]))  # C1..C19

    t_cyc, duty, profile_rows = load_profile(component_ids)
    T = args.N * t_cyc

    # 单元件：λ_eff=λ*duty；R(T)=exp(-λ_eff*T)
    lam_eff = {}
    R_map = {}
    for cid in component_ids:
        name, lam = comps[cid]
        lam_eff[cid] = lam * duty[cid]
        R_map[cid] = R_exp(lam_eff[cid], T)

    model = parse_model()

    # 检查所有元件都被RBD结构引用
    def collect_rbd_components(node):
        if isinstance(node, str):
            return {node}
        if isinstance(node, dict):
            if "series" in node:
                s = set()
                for x in node["series"]:
                    s |= collect_rbd_components(x)
                return s
            if "parallel" in node:
                s = set()
                for x in node["parallel"]:
                    s |= collect_rbd_components(x)
                return s
        return set()

    rbd_cids = collect_rbd_components(model)
    missing = [cid for cid in component_ids if cid not in rbd_cids]
    if missing:
        raise ValueError(f"RBD结构未包含所有元件，缺少: {missing}\n请检查model.json，确保所有元件都被引用。")

    R_sys = eval_node(model, R_map)

    #  计算各个子系统的可靠度
    subsystem_R = calculate_subsystem_reliability(R_map)

    # ---- sanity checks ----
    # 1) 去冗余应变差
    model_stripped = strip_parallel(model)
    R_sys_stripped = eval_node(model_stripped, R_map)
    check1_ok = (R_sys_stripped <= R_sys + 1e-12)

    # 2) 缩短任务时间应变好（T/2）
    R_map_half = {cid: R_exp(lam_eff[cid], T / 2.0) for cid in component_ids}
    R_sys_half = eval_node(model, R_map_half)
    check2_ok = (R_sys_half >= R_sys - 1e-12)

    if not check1_ok or not check2_ok:
        raise RuntimeError(
            "Sanity check 未通过：\n"
            f"- 去冗余应变差：{check1_ok}（R_noRed={R_sys_stripped:.6f}, R={R_sys:.6f})\n"
            f"- 缩短任务时间应变好：{check2_ok}（R_halfT={R_sys_half:.6f}, R={R_sys:.6f})\n"
            "请检查 mission_profile.csv 与 model.json 的合理性（边界/并联/单位/占空比）。"
        )

    weak_id = min(component_ids, key=lambda k: R_map[k])
    weak_name, _ = comps[weak_id]

    # ---- output markdown ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# {EXPERIMENT_NAME}")
    lines.append(f"{args.student_id}，{args.student_name}")
    lines.append("")

    lines.append("## 1. 任务与剖面参数")
    lines.append(f"- 单循环时长（由 mission_profile 计算）：t_cyc = {t_cyc:.3f} h")
    lines.append(f"- 循环次数：N = {args.N}")
    lines.append(f"- 班次任务时间：T = N * t_cyc = {T:.3f} h")
    lines.append("")

    lines.append("## 2. 任务剖面（阶段时长）")
    lines.append("| 阶段 | duration_h |")
    lines.append("|---|---:|")
    for r in profile_rows:
        lines.append(f"| {r['phase']} | {r['duration_h']:.3f} |")
    lines.append("")

    lines.append("## 3.  RBD（model.json）")
    lines.append("```json")
    # 只展示 model 段，避免把 hints 一起打印
    lines.append(json.dumps(model, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## 4. 单元件参数与可靠度（R_i = exp(-λ_eff * T)）")
    lines.append("| 编号 | 元件 | λ(1/h) | duty | λ_eff(1/h) | R(T) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cid in component_ids:
        name, lam = comps[cid]
        lines.append(f"| {cid} | {name} | {lam:.2e} | {duty[cid]:.3f} | {lam_eff[cid]:.2e} | {R_map[cid]:.6f} |")
    lines.append("")

    lines.append("## 5. 子系统可靠度 (Subsystem Reliability)")
    lines.append("| 子系统名称 | 子系统可靠度 R_sub(T) |")
    lines.append("|---|---:|")
    for sub_name, r_val in subsystem_R.items():
        lines.append(f"| {sub_name} | {r_val:.6f} |")
    lines.append("")

    lines.append("## 6. 系统任务可靠度")
    lines.append(f"- R_sys(T) = {R_sys:.6f}")
    lines.append("")

    lines.append("## 7. 薄弱环节（最小 R(T)）")
    lines.append(f"- {weak_id} {weak_name}：R(T) = {R_map[weak_id]:.6f}（λ_eff={lam_eff[weak_id]:.2e} 1/h）")
    lines.append("")

    lines.append("## 8. Sanity checks（必须通过）")
    lines.append(f"- 去冗余应变差：PASS（R_noRed={R_sys_stripped:.6f} ≤ R={R_sys:.6f}）")
    lines.append(f"- 缩短任务时间应变好：PASS（R_halfT={R_sys_half:.6f} ≥ R={R_sys:.6f}）")
    lines.append("")

    lines.append(f"> 报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 保留自定义区块内容（如果已存在）
    custom_block_title = "## 9. 学生自定义补充区"

    # 填入写好的内容
    custom_block_content = f"""{custom_block_title}

    ### 任务 1：建模与公式

     1. 任务时间与有效失效率模型
     根据任务剖面，完整搬运任务包含 60 个循环，每个循环时长 t_cyc = 0.1h，故总任务时间 T 为： T = N ✖ t_cyc = 60 ✖ 0.1 = 6.0h

     由于部分元件在任务循环中并非全程满载运行（例如起升元件、小车元件等），需引入占空比（duty）计算有效失效率 λ_eff：λ_effi =λi ✖ duty.  
     即λ_eff1 = λ1 ✖ duty1 = λ1 ✖ duty = 2.0e-5 ✖ 1 = 2.0e-5 
      λ_eff2 = λ2 ✖ duty2 = λ2 ✖ duty = 1.0e-5 ✖ 1 = 1.0e-5
      λ_eff3 = λ3 ✖ duty3 = λ3 ✖ duty = 8.0e-5 ✖ 1 = 8.0e-5
      λ_eff4 = λ4 ✖ duty4 = λ4 ✖ duty = 8.0e-5 ✖ 1 = 8.0e-5
      λ_eff5 = λ5 ✖ duty5 = λ5 ✖ duty = 3.0e-5 ✖ 1 = 3.0e-5
      λ_eff6 = λ6 ✖ duty6 = λ6 ✖ duty = 5.0e-5 ✖ 1 = 5.0e-5
      λ_eff7 = λ7 ✖ duty7 = λ7 ✖ duty = 4.0e-5 ✖ 1 = 4.0e-5
      λ_eff8 = λ8 ✖ duty8 = λ8 ✖ d_H = 1.2e-4 ✖ 0.35 = 4.20e-05
      λ_eff9 = λ9 ✖ duty9 = λ9 ✖ d_H = 6.0e-5 ✖ 0.35 = 2.10e-05
      λ_eff10 = λ10 ✖ duty10 = λ10 ✖ d_H = 4.0e-5 ✖ 0.35 = 1.40e-05
      λ_eff11 = λ11 ✖ duty11 = λ11 ✖ d_H = 3.5e-5 ✖ 0.35 = 1.22e-05
      λ_eff12 = λ12 ✖ duty12 = λ12 ✖ d_T = 1.0e-4 ✖ 0.40 = 4.50e-05
      λ_eff13 = λ13 ✖ duty13 = λ13 ✖ d_T = 5.0e-5 ✖ 0.40 = 2.25e-05
      λ_eff14 = λ14 ✖ duty14 = λ14 ✖ d_T = 2.5e-5 ✖ 0.40 = 1.12e-05
      λ_eff15 = λ15 ✖ duty15 = λ15 ✖ duty = 7.0e-5 ✖ 1 = 7.0e-5
      λ_eff16 = λ16 ✖ duty16 = λ16 ✖ duty = 7.0e-5 ✖ 1 = 7.0e-5
      λ_eff17 = λ17 ✖ duty17 = λ17 ✖ duty = 6.0e-5 ✖ 1 = 6.0e-5 
      λ_eff18 = λ18 ✖ duty18 = λ17 ✖ duty = 6.0e-5 ✖ 1 = 6.0e-5 
      λ_eff19 = λ19 ✖ duty19 = λ19 ✖ d_G = 6.0e-5 ✖ 0.2 = 1.20e-05

     2. 元件可靠度
     假设各元件寿命服从指数分布，则单一元件在任务时间 T内的可靠度R_i(T)为：
     R_i(T) = exp(-λ_effi ✖ T)
     即R_1(6)=0.999880，R_2(6)=0.999940，R_3(6)=0.999520，R_4(6)=0.999520
       R_5(6)=0.999820，R_6(6)=0.999700，R_7(6)=0.999760，R_8(6)=0.999748
       R_9(6)=0.999874，R_10(6)=0.999916，R_11(6)=0.999927，R_12(6)=0.999730
       R_13(6)=0.999865，R_14(6)=0.999933，R_15(6)=0.999580，R_16(6)=0.999580
       R_17(6)=0.999640，R_18(6)=0.999640，R_19(6)=0.999928
     
     3. 1oo2 并联可靠度
     对于采用 1oo2（双重冗余）并联结构的子系统（如整流模块 C3/C4，起升上限位 C15/C16，小车编码器 C17/C18），只要其中一个元件正常工作即可算作子系统成功，其并联可靠度计算公式为：
     R = 1 - (1 - R_A(T))(1 - R_B(T))
     即整流模块 C3/C4：R=1-(1-R3)(1-R4)=0.9999997696
     起升上限位 C15/C16: R=1-(1-R15)(1-R16)=0.9999998236
     小车编码器 C17/C18: R=1-(1-R17)(1-R18)=0.9999998074
     
     4. 系统总体可靠度表达式
     根据本实验的 RBD 结构（主干串联，局部包含 3 组并联，2 组子串联），由于子串联在数学上等同于直接串联，系统总体任务可靠度 R_sys(T) 的完整表达式为：
     R_sys(T)= R1·R2·[1-(1-R3)(1-R4)]·R5·R6·R7·(R8·R9·R10·R11)·(R12·R13·R14)·[1-(1-R15)(1-R16)]·[1-(1-R17)(1-R18)]·R19 

    ### 任务 3：工程解释

     1. 独立核验（Sanity checks）结果验证与解释
     1）去冗余应变差 ：通过将系统中的并联冗余（C4, C16, C18）人为移除，使整个系统退化为纯串联系统，测得系统可靠度 R_noRed=0.996765，低于原始系统的 R_sys = 0.998021。
        工程解释：结果符合实验预期（把所有并联节点替换为单支路后，系统可靠度应不高于原结果）。因为并联冗余为系统提供了额外的“容错路径”，即便主元件失效，备用元件仍能保障任务完成。去掉冗余后，所有模块都变成了系统的“单点故障源”，系统发生失效的概率必然上升，整体可靠度随之下降。
     2）缩短任务时间应变好 ：将运行班次任务时间减半后，测得系统可靠度 R_halfT=0.999010，高于原始系统的 R_sys = 0.998021。
        工程解释：结果符合实验预期（把总任务时间减半后，系统可靠度应更高），因为将运行班次任务时间减半后，设备暴露在随机失效中的时间基数变小，系统整体可靠度上升。
     2. 薄弱环节分析
     绝对可靠度最低的个体：根据计算表，C3 整流模块A 及 C4 整流模块B 的有效失效率最高（λ_eff=8.0e-05），其个体的单次任务可靠度最低（R_i(T) = 0.999520）。
     系统级薄弱环节（单点故障瓶颈）：虽然 C3 和 C4 个体可靠度最低，但因为它们互为并联冗余，其子系统整体可靠度极高（R=1-(1-0.99952）^2=0.9999997）。从整个系统来看，真正的薄弱环节是串联主干上可靠度最低的元件，即 C6 PLC控制器（R=0.999700）。如果它损坏，系统将直接停机。

     3. 工程改进思路
     改进方案：对 C6 PLC控制器 增加双机热备（1oo2并联冗余）；或者选购工业等级更高、基础失效率λ更低的 PLC 控制器与变频器。
     预期影响方向：当前 C6 作为串联核心节点，由于缺乏冗余，直接拉低了整体的 R_sys(T)。增加 PLC 热备冗余后，可以消除这个最大的单点故障风险；预期系统的整体任务可靠度 R_sys(T) 将会实现跨越式提升（有望突破 0.999 的高可靠性大关），系统容错能力大幅增强。

    ### 10. AI 使用记录
    - 是否使用 AI: 是
    - 问题 1：什么是 RBD？
      - AI 解答：RBD 的全称是 Reliability Block Diagram，中文叫 可靠性框图。它是可靠性工程（Reliability Engineering）中非常核心的一个工具。简单来说，它是一种用图形来描述系统中各个零件如何影响整个系统状态的逻辑模型。RBD图画的不是物理连接线，而是“逻辑关系线”。主要有串联结构和并联结构。
    - 问题 2：如何计算串联结构和并联结构的可靠度？
      - AI 解答：串联：R_sys=R1✖R2✖R3…… ；并联：R_sys=1-（1-R_A)(1-R_B)
    - 问题 3：为我检查代码是否有错？
      - AI 解答：定位了之前因多行字符串嵌套导致的 SyntaxError 报错，并修复了 Python 输出报告时的语法格式。
    - 核验与修正：我人工核验了 AI 提供的串并联概率公式，确认公式完全正确；并将存在语法报错的 f-string 文本字符串彻底修正。
    """

    # 强制将预置的内容赋给 custom_block（忽略旧文件中可能存在的空白占位）
    custom_block = custom_block_content

    # 原逻辑保留：如果已经有用户自行补充的额外内容，保留并在最前面补上Title
    if OUT_PATH.exists():
        old = OUT_PATH.read_text(encoding="utf-8")
        if custom_block_title in old:
            extracted_old_block = old.split(custom_block_title, 1)[1].lstrip('\n')
            # 只有当旧文件里确实有其他有意义的内容时，才优先保留旧内容，否则用我们写好的
            if "任务 1：建模与公式" not in extracted_old_block and len(extracted_old_block.strip()) > 100:
                custom_block = f"{custom_block_title}\n" + extracted_old_block

    lines.append(custom_block)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成：{OUT_PATH}")
    print(f"R_sys(T={T:.3f}h) = {R_sys:.6f}")


if __name__ == "__main__":
    main()