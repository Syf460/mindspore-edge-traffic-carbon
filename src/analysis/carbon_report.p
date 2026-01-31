import re
import json
from typing import Dict, Any, Optional

def _strip_think(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

def _one_sentence_cn(text: str, max_len: int = 50) -> str:
    if not text:
        return ""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    s = lines[-1] if lines else text.strip()
    s = re.sub(r"\s+", " ", s).strip()
    parts = re.split(r"[。；;！？!?]", s)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        s = parts[0]
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s

def _calc_total_co2(road_km: float, car: int, truck: int,
                    car_factor: float, truck_factor: float) -> float:
    return road_km * (car * car_factor + truck * truck_factor)

def _fallback_summary(road_km: float, car: int, truck: int, total_co2: float) -> str:
    total = car + truck
    if total <= 0:
        return "本路段未检测到车辆，暂无法核算碳排放。"
    return f"{road_km:g}公里路段检测{total}辆车，预计排放约{total_co2:.2f}千克CO₂。"

def make_carbon_report_text(
    road_km: float,
    car: int,
    truck: int,
    car_factor: float,
    truck_factor: float,
    llm_output: str = "",
) -> Dict[str, Any]:
    total_co2 = _calc_total_co2(road_km, car, truck, car_factor, truck_factor)

    summary = _one_sentence_cn(_strip_think(llm_output), 50)
    if not summary:
        summary = _fallback_summary(road_km, car, truck, total_co2)

    lines = []
    lines.append("你是一位环保智能助手沫蕊。请根据以下数据，生成一段简洁、自然、具说明性的汇报句子：\n")
    lines.append("    —— 沫芯·碳排放分析报告 ——")
    lines.append(f"路段长度： {road_km:g} km")
    lines.append(f"检测结果： 汽车 {car} 辆， 货车 {truck} 辆")
    lines.append(f"排放系数： 汽车 {car_factor:g} kg/km， 货车 {truck_factor:g} kg/km")
    lines.append("——————————————————————")
    lines.append(f"总碳排放 ≈ {total_co2:.2f} kg CO₂\n")
    lines.append("要求：用中文简短总结一句话，不超过50字。")
    lines.append(f"\n{summary}")

    return {
        "total_co2_kg": round(total_co2, 2),
        "summary": summary,
        "report_text": "\n".join(lines)
    }

def print_carbon_report(
    stats: Dict[str, Any],
    road_km: float,
    car_factor: float,
    truck_factor: float,
    llm_output: str = "",
    write_back_json_path: Optional[str] = None
) -> Dict[str, Any]:
    car = int(stats.get("car", 0))
    truck = int(stats.get("truck", 0))

    report = make_carbon_report_text(
        road_km=road_km,
        car=car,
        truck=truck,
        car_factor=car_factor,
        truck_factor=truck_factor,
        llm_output=llm_output
    )

    print(report["report_text"])

    if write_back_json_path:
        merged = dict(stats)
        merged["road_km"] = road_km
        merged["emission_factor_kg_per_km"] = {"car": car_factor, "truck": truck_factor}
        merged["total_co2_kg"] = report["total_co2_kg"]
        merged["llm_summary"] = report["summary"]
        with open(write_back_json_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    return report
