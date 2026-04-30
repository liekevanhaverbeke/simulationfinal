from pathlib import Path
import math
import csv
import shutil

out_dir = Path("generated_input_files")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
CAPACITY = [32, 32, 32, 16, 32, 16]
HALF_DAY_COLS = {3, 5}

def create_base_schedule():
    schedule = [[1 for _ in range(6)] for _ in range(32)]
    for col in HALF_DAY_COLS:
        for row in range(16, 32):
            schedule[row][col] = 0
    return schedule

def allocate_urgent_slots_over_days(total_urgent):
    total_capacity = sum(CAPACITY)
    quotas = [total_urgent * cap / total_capacity for cap in CAPACITY]
    allocation = [math.floor(q) for q in quotas]
    remaining = total_urgent - sum(allocation)
    remainders = [(quotas[i] - allocation[i], i) for i in range(6)]
    remainders.sort(key=lambda x: (-x[0], x[1]))
    max_per_day = [4, 4, 4, 2, 4, 2]
    idx = 0
    while remaining > 0:
        _, day_idx = remainders[idx % len(remainders)]
        if allocation[day_idx] < max_per_day[day_idx]:
            allocation[day_idx] += 1
            remaining -= 1
        idx += 1
    return allocation

S1_POSITIONS_FULL = {1: [16], 2: [16, 32], 3: [15, 16, 32], 4: [15, 16, 31, 32]}
S1_POSITIONS_HALF = {1: [16], 2: [15, 16]}
S2_POSITIONS_FULL = {1: [16], 2: [11, 22], 3: [8, 16, 24], 4: [6, 13, 20, 26]}
S2_POSITIONS_HALF = {1: [8], 2: [5, 11], 3: [4, 9, 13], 4: [3, 7, 11, 14]}
S3_POSITIONS_FULL = {1: [7], 2: [7, 14], 3: [7, 14, 21], 4: [7, 14, 21, 28]}
S3_POSITIONS_HALF = {1: [7], 2: [7, 14]}

POSITIONS = {
    "S1": (S1_POSITIONS_FULL, S1_POSITIONS_HALF),
    "S2": (S2_POSITIONS_FULL, S2_POSITIONS_HALF),
    "S3": (S3_POSITIONS_FULL, S3_POSITIONS_HALF),
}

def generate_schedule(strategy, total_urgent):
    schedule = create_base_schedule()
    allocation = allocate_urgent_slots_over_days(total_urgent)
    full_positions, half_positions = POSITIONS[strategy]
    for col, n_slots_day in enumerate(allocation):
        day_positions = half_positions[n_slots_day] if col in HALF_DAY_COLS else full_positions[n_slots_day]
        for slot_nr in day_positions:
            schedule[slot_nr - 1][col] = 2
    return schedule, allocation

def write_schedule(schedule, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for row in schedule:
            f.write("\t".join(map(str, row)) + "\n")

summary_rows = []
for strategy in ["S1", "S2", "S3"]:
    for total_urgent in range(10, 21):
        schedule, allocation = generate_schedule(strategy, total_urgent)
        assert sum(value == 2 for row in schedule for value in row) == total_urgent
        filename = f"input-{strategy}-{total_urgent}.txt"
        write_schedule(schedule, out_dir / filename)
        summary_rows.append({
            "file": filename,
            "strategy": strategy,
            "urgent_slots_total": total_urgent,
            "Mon": allocation[0],
            "Tue": allocation[1],
            "Wed": allocation[2],
            "Thu": allocation[3],
            "Fri": allocation[4],
            "Sat": allocation[5],
        })

with open(out_dir / "input_generation_summary.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "strategy", "urgent_slots_total", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Generated {len(summary_rows)} input files in {out_dir.resolve()}")
