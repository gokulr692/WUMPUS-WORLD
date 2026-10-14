"""
Wumpus World - Knowledge-Based Agent using Propositional Logic
Course : 24AI301 Artificial Intelligence and Data Science
Student: M. Gokul Sai Prasad Reddy (111725201305)
"""
import random
import sys
from itertools import product
from collections import deque

N = 4  # 4 x 4 cave


def neighbours(c):
    x, y = c
    cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [(a, b) for a, b in cand if 1 <= a <= N and 1 <= b <= N]


# ---------------------------------------------------------------- ENVIRONMENT
class WumpusWorld:
    def __init__(self, seed=None, pit_prob=0.2):
        rnd = random.Random(seed)
        cells = [(x, y) for x in range(1, N + 1) for y in range(1, N + 1)]
        cells.remove((1, 1))
        self.pits = {c for c in cells if rnd.random() < pit_prob}
        free = [c for c in cells if c not in self.pits] or cells
        self.wumpus = rnd.choice(free)
        self.gold = rnd.choice([c for c in cells if c not in self.pits])
        self.wumpus_alive = True

    def percept(self, c):
        stench = self.wumpus_alive and (c == self.wumpus or self.wumpus in neighbours(c))
        breeze = any(n in self.pits for n in neighbours(c))
        glitter = (c == self.gold)
        return {"stench": stench, "breeze": breeze, "glitter": glitter}

    def shoot(self, frm, target):
        """Arrow flies in a straight line from 'frm' towards 'target'."""
        dx = (target[0] > frm[0]) - (target[0] < frm[0])
        dy = (target[1] > frm[1]) - (target[1] < frm[1])
        x, y = frm
        while 1 <= x <= N and 1 <= y <= N:
            if (x, y) == self.wumpus and self.wumpus_alive:
                self.wumpus_alive = False
                return True
            x, y = x + dx, y + dy
        return False


# ---------------------------------------------------------------------- AGENT
class KBAgent:
    def __init__(self):
        self.pos = (1, 1)
        self.visited = {(1, 1)}
        self.percepts = {}          # cell -> percept dict   (the Knowledge Base)
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_dead = False
        self.score = 0
        self.path = [(1, 1)]

    # ---- TELL -----------------------------------------------------------
    def tell(self, cell, p):
        self.percepts[cell] = p

    # ---- ASK (model checking over propositional symbols P_xy, W_xy) ------
    def pit_models(self):
        frontier = sorted({n for v in self.visited for n in neighbours(v)} - self.visited)
        models = []
        for bits in product([False, True], repeat=len(frontier)):
            pit = dict(zip(frontier, bits))
            ok = True
            for v, p in self.percepts.items():
                if p["breeze"] != any(pit.get(n, False) for n in neighbours(v)):
                    ok = False
                    break
            if ok:
                models.append(pit)
        return frontier, models

    def wumpus_candidates(self):
        if self.wumpus_dead:
            return set()
        cand = set()
        for c in [(x, y) for x in range(1, N + 1) for y in range(1, N + 1)]:
            if c in self.visited:
                continue
            ok = True
            for v, p in self.percepts.items():
                near = (c == v) or (c in neighbours(v))
                if p["stench"] != near:
                    ok = False
                    break
            if ok:
                cand.add(c)
        return cand

    def analyse(self):
        frontier, models = self.pit_models()
        wc = self.wumpus_candidates()
        safe, risk = set(), {}
        for c in frontier:
            pit_cnt = sum(m[c] for m in models)
            if pit_cnt == 0 and c not in wc:
                safe.add(c)
            else:
                risk[c] = pit_cnt / max(len(models), 1) + (0.3 if c in wc else 0)
        return safe, risk, wc

    # ---- navigation ------------------------------------------------------
    def bfs(self, goal, allowed):
        q, prev = deque([self.pos]), {self.pos: None}
        while q:
            cur = q.popleft()
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            for n in neighbours(cur):
                if n in allowed and n not in prev:
                    prev[n] = cur
                    q.append(n)
        return None


# ------------------------------------------------------------------ SIMULATION
def run(seed, verbose=True, max_steps=100):
    w = WumpusWorld(seed)
    a = KBAgent()
    log = []

    def say(msg):
        log.append(msg)
        if verbose:
            print(msg)

    def sense(cell):
        p = w.percept(cell)
        a.tell(cell, p)
        found = [k.capitalize() for k, v in p.items() if v]
        say(f"  Percepts at {cell}: {', '.join(found) if found else 'None'}")
        return p

    def move(cell):
        a.score -= 1
        a.pos = cell
        a.path.append(cell)

    say(f"Hidden world -> Pits: {sorted(w.pits)}  Wumpus: {w.wumpus}  Gold: {w.gold}")
    say("=" * 66)
    p = sense((1, 1))
    outcome = "Agent stopped without finding the gold"

    for step in range(1, max_steps + 1):
        if p["glitter"] and not a.has_gold:
            a.has_gold = True
            a.score += 1000
            say(f"Step {step}: GRAB gold at {a.pos}  (+1000)")
        if a.has_gold:
            back = a.bfs((1, 1), a.visited)
            say(f"Step {step}: Returning home via {back}")
            for c in back[1:]:
                move(c)
            a.score -= 1  # CLIMB
            say("CLIMB out of the cave at (1,1)")
            outcome = "SUCCESS - gold collected and agent escaped"
            break

        safe, risk, wc = a.analyse()
        targets = [c for c in safe if c not in a.visited]
        wtxt = (str(sorted(wc)) if len(wc) <= 3 else f"{len(wc)} cells")
        say(f"Step {step}: At {a.pos} | SAFE: {sorted(targets)} | Wumpus possible: {wtxt}")

        if targets:
            goal = min(targets, key=lambda c: len(a.bfs(c, a.visited | {c}) or range(99)))
        elif len(wc) == 1 and a.has_arrow:
            wpos = next(iter(wc))
            shooters = [v for v in a.visited if v[0] == wpos[0] or v[1] == wpos[1]]
            if shooters:
                spot = min(shooters, key=lambda c: len(a.bfs(c, a.visited) or range(99)))
                for c in a.bfs(spot, a.visited)[1:]:
                    move(c)
                a.has_arrow = False
                a.score -= 10
                hit = w.shoot(a.pos, wpos)
                say(f"  SHOOT arrow from {a.pos} towards {wpos} (-10) -> "
                    f"{'SCREAM! Wumpus killed' if hit else 'missed'}")
                if hit:
                    a.wumpus_dead = True
                for c in list(a.visited):
                    a.percepts[c] = dict(a.percepts[c], stench=False)
                continue
            goal = None
        else:
            goal = None

        if goal is None:
            if not risk:
                outcome = "No safe or unexplored cells left - agent gives up"
                break
            goal = min(risk, key=risk.get)
            say(f"  No provably safe move -> taking calculated risk at {goal} "
                f"(est. danger {risk[goal]:.2f})")

        route = a.bfs(goal, a.visited | {goal})
        for c in route[1:]:
            move(c)
        say(f"  MOVE {route[0]} -> {goal}  (score {a.score})")

        if goal in w.pits:
            a.score -= 1000
            outcome = f"DEAD - fell into a pit at {goal}"
            say(f"  !!! Agent fell into a PIT at {goal} (-1000)")
            break
        if goal == w.wumpus and w.wumpus_alive:
            a.score -= 1000
            outcome = f"DEAD - eaten by the Wumpus at {goal}"
            say(f"  !!! Agent was eaten by the WUMPUS at {goal} (-1000)")
            break
        a.visited.add(goal)
        p = sense(goal)

    say("=" * 66)
    say(f"RESULT : {outcome}")
    say(f"SCORE  : {a.score}")
    say(f"CELLS EXPLORED: {len(a.visited)}/16 | PATH LENGTH: {len(a.path) - 1}")
    return w, a, outcome, log


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    run(seed)
