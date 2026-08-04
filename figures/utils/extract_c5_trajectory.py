"""Extract per-round per-class IoU from a training log."""
import re
import sys
import json

def parse_log(path):
    """Parse rows like: ' 20  |   0.5130   |   0.5814   |  0.5472  | [0.915 0.754 0.872 0.430 0.091 0.221]'."""
    trajectory = []  # list of (round, test1_miou, test2_miou, avg_miou, [c0..c5])
    row_re = re.compile(r'^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\[([^\]]+)\]')
    with open(path) as f:
        for line in f:
            m = row_re.match(line)
            if m:
                rnd, t1, t2, avg, cls = m.groups()
                ious = [float(x) for x in cls.split()]
                trajectory.append({
                    'round': int(rnd),
                    'test1': float(t1),
                    'test2': float(t2),
                    'avg': float(avg),
                    'per_class': ious,
                })
    return trajectory

if __name__ == '__main__':
    traj = parse_log(sys.argv[1])
    print(json.dumps(traj, indent=2))
