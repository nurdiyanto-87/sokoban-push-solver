// Sokoban solver: room-decomposed push-optimal A*.
//
// Idea: every level is a 1-wide hallway with 1-4 rooms off it (3-4 boxes
// each). A box pushed out of its room through the door can never come back,
// so each room is solved on its own, from its door, and the room solutions
// are stitched together with plain player walks along the hallway.
// Inside a room we run a push-based A* over (sorted box positions,
// canonical player cell), with an admissible heuristic (min-cost assignment
// of boxes to goals over single-box push distances), dead-square pruning
// (reverse push BFS from the goals), corner and 2x2 frozen-block pruning.
// Search is budgeted by node expansions, never by the clock, so two runs
// produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DEBUG = process.env.MAKEFUN_DEBUG === '1';

// Directions: 0 = up, 1 = down, 2 = left, 3 = right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const DL = ['u', 'd', 'l', 'r'];
const OPP = [1, 0, 3, 2];
const INF = 1e9;

// Per-room node budgets (expansions). Deterministic: counts, not time.
const BUDGET_OPT = 400000;
const BUDGET_WEIGHTED = 400000;
const BUDGET_GREEDY = 400000;

let expansionsTotal = 0;

type Level = {
  w: number; h: number;
  wall: Uint8Array; goal: Uint8Array; box: Uint8Array;
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const box = new Uint8Array(w * h);
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = x < row.length ? row[x] : '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') box[i] = 1;
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goal, box, player };
}

type RoomInfo = {
  cells: number[]; // global indices of room floor cells
  door: number;    // global index of the door cell (in the room)
  entry: number;   // global index of the hallway cell outside the door
};

// Split the level into rooms: connected floor components excluding the
// hallway row (the row the player starts in). Each room has one door cell.
function findRooms(L: Level): RoomInfo[] {
  const { w, h, wall, box, goal, player } = L;
  const py = (player / w) | 0;
  const seen = new Uint8Array(w * h);
  const rooms: RoomInfo[] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (wall[i] || y === py || seen[i]) continue;
      const cells: number[] = [];
      const q = [i];
      seen[i] = 1;
      while (q.length > 0) {
        const c = q.pop() as number;
        cells.push(c);
        const cx = c % w;
        const cy = (c / w) | 0;
        for (let d = 0; d < 4; d++) {
          const nx = cx + DX[d];
          const ny = cy + DY[d];
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const ni = ny * w + nx;
          if (wall[ni] || ny === py || seen[ni]) continue;
          seen[ni] = 1;
          q.push(ni);
        }
      }
      cells.sort((a, b) => a - b);
      let nb = 0;
      for (const c of cells) {
        if (box[c]) nb++;
      }
      if (nb === 0) continue;
      // Door: a room cell adjacent to a hallway-row floor cell.
      let door = -1;
      let entry = -1;
      for (const c of cells) {
        const cx = c % w;
        const cy = (c / w) | 0;
        for (let d = 0; d < 4; d++) {
          const nx = cx + DX[d];
          const ny = cy + DY[d];
          if (ny !== py || nx < 0 || nx >= w) continue;
          const ni = ny * w + nx;
          if (!wall[ni]) {
            door = c;
            entry = ni;
          }
        }
        if (door >= 0) break;
      }
      if (door < 0) continue; // no hallway access; cannot be solved
      rooms.push({ cells, door, entry });
    }
  }
  return rooms;
}

// Breadth-first walk (no pushes) from `from` to `to` over floor cells that
// hold no box. Returns the move string, or null when unreachable.
function bfsPath(L: Level, from: number, to: number, boxAt: Uint8Array): string | null {
  if (from === to) return '';
  const { w, h, wall } = L;
  const n = w * h;
  const prev = new Int32Array(n).fill(-1);
  const prevDir = new Int8Array(n).fill(-1);
  const q: number[] = [from];
  prev[from] = from;
  let head = 0;
  while (head < q.length) {
    const c = q[head++];
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const ni = ny * w + nx;
      if (wall[ni] || boxAt[ni] || prev[ni] >= 0) continue;
      prev[ni] = c;
      prevDir[ni] = d;
      if (ni === to) {
        let s = '';
        let cur = ni;
        while (cur !== from) {
          s = DL[prevDir[cur]] + s;
          cur = prev[cur];
        }
        return s;
      }
      q.push(ni);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Per-room solver context (local cell indices 0..C-1 over room floor + entry).
// ---------------------------------------------------------------------------
type RoomCtx = {
  C: number;
  gx: Int32Array; gy: Int32Array;
  nbr: Int32Array; // C*4, local neighbour index or -1
  isGoal: Uint8Array;
  goals: number[];
  dist: Int32Array; // goals.length * C; min pushes box->goal (INF = unreachable)
  deadCell: Uint8Array;
  cornerDead: Uint8Array;
  blocks: number[][]; // 2x2 squares, entries are local indices or -1 (wall)
  boxes0: number[]; // initial box local indices, sorted
  entry: number; // local index of the hallway entry cell
  perms: number[][]; // all permutations of goal indices (for the assignment)
};

function buildRoomCtx(L: Level, room: RoomInfo): RoomCtx {
  const { w, wall, goal, box } = L;
  // Allowed cells: room floor + the single hallway entry cell.
  const allowed = new Set<number>(room.cells);
  allowed.add(room.entry);
  const list = [...allowed].sort((a, b) => a - b);
  const C = list.length;
  const toLocal = new Int32Array(w * L.h).fill(-1);
  const gx = new Int32Array(C);
  const gy = new Int32Array(C);
  list.forEach((g, li) => {
    toLocal[g] = li;
    gx[li] = g % w;
    gy[li] = (g / w) | 0;
  });
  const nbr = new Int32Array(C * 4).fill(-1);
  for (let li = 0; li < C; li++) {
    for (let d = 0; d < 4; d++) {
      const nx = gx[li] + DX[d];
      const ny = gy[li] + DY[d];
      if (nx < 0 || ny < 0 || nx >= w || ny >= L.h) continue;
      const ni = ny * w + nx;
      if (!wall[ni] && toLocal[ni] >= 0) nbr[li * 4 + d] = toLocal[ni];
    }
  }
  const isGoal = new Uint8Array(C);
  const goals: number[] = [];
  const boxes0: number[] = [];
  for (let li = 0; li < C; li++) {
    const g = list[li];
    if (goal[g]) {
      isGoal[li] = 1;
      goals.push(li);
    }
    if (box[g]) boxes0.push(li);
  }
  boxes0.sort((a, b) => a - b);
  // Single-box push distances: reverse BFS from each goal. A reverse step
  // moves the box from c to the adjacent cell n = c - dir, and is valid only
  // when the player's stance cell c - 2*dir is floor (so the forward push
  // that produced it could have happened). This over-approximates true
  // reachability, so INF here means the cell is truly dead.
  const G = goals.length;
  const dist = new Int32Array(G * C).fill(INF);
  for (let gi = 0; gi < G; gi++) {
    const base = gi * C;
    const start = goals[gi];
    dist[base + start] = 0;
    const q = [start];
    let head = 0;
    while (head < q.length) {
      const c = q[head++];
      const dc = dist[base + c];
      for (let d = 0; d < 4; d++) {
        const n = nbr[c * 4 + OPP[d]];
        if (n < 0) continue;
        const stance = nbr[n * 4 + OPP[d]];
        if (stance < 0) continue;
        if (dist[base + n] !== INF) continue;
        dist[base + n] = dc + 1;
        q.push(n);
      }
    }
  }
  const deadCell = new Uint8Array(C);
  for (let c = 0; c < C; c++) {
    let live = false;
    for (let gi = 0; gi < G; gi++) {
      if (dist[gi * C + c] < INF) {
        live = true;
        break;
      }
    }
    if (!live) deadCell[c] = 1;
  }
  // Corner cells: two orthogonal neighbours are walls. A box there (not on
  // a goal) can never move: both pushes need the player to stand in a wall.
  const cornerDead = new Uint8Array(C);
  for (let c = 0; c < C; c++) {
    if (isGoal[c]) continue;
    const up = nbr[c * 4 + 0] < 0;
    const dn = nbr[c * 4 + 1] < 0;
    const lf = nbr[c * 4 + 2] < 0;
    const rt = nbr[c * 4 + 3] < 0;
    if ((up && lf) || (up && rt) || (dn && lf) || (dn && rt)) cornerDead[c] = 1;
  }
  // 2x2 blocks: every 2x2 square of the board that touches the room. A block
  // whose floor cells are all occupied by boxes, one of them off-goal, is a
  // frozen deadlock (no box in it can ever move).
  const blocks: number[][] = [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let li = 0; li < C; li++) {
    if (gx[li] < minX) minX = gx[li];
    if (gx[li] > maxX) maxX = gx[li];
    if (gy[li] < minY) minY = gy[li];
    if (gy[li] > maxY) maxY = gy[li];
  }
  for (let y = minY - 1; y <= maxY; y++) {
    for (let x = minX - 1; x <= maxX; x++) {
      const sq: number[] = [];
      let anyFloor = false;
      for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
        const nx = x + ox;
        const ny = y + oy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= L.h || wall[ny * w + nx]) {
          sq.push(-1);
        } else {
          const li = toLocal[ny * w + nx];
          sq.push(li);
          if (li >= 0) anyFloor = true;
        }
      }
      if (anyFloor) blocks.push(sq);
    }
  }
  // All permutations of goal indices (<= 4 goals -> <= 24 permutations).
  const perms: number[][] = [];
  const idx = goals.map((_, i) => i);
  const gen = (a: number[], k: number): void => {
    if (k === 1) {
      perms.push(a.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      gen(a, k - 1);
      const j = k % 2 === 0 ? i : 0;
      const t = a[j];
      a[j] = a[k - 1];
      a[k - 1] = t;
    }
  };
  gen(idx, idx.length);
  return {
    C, gx, gy, nbr, isGoal, goals, dist, deadCell, cornerDead, blocks,
    boxes0, entry: toLocal[room.entry], perms,
  };
}

// Flood fill from `start` over allowed cells avoiding `boxSet`; returns the
// visited mask and the smallest cell index reached (canonical player cell).
function floodFill(ctx: RoomCtx, start: number, boxSet: Uint8Array): { mark: Uint8Array; min: number } {
  const mark = new Uint8Array(ctx.C);
  const q = [start];
  mark[start] = 1;
  let min = start;
  let head = 0;
  while (head < q.length) {
    const c = q[head++];
    if (c < min) min = c;
    for (let d = 0; d < 4; d++) {
      const n = ctx.nbr[c * 4 + d];
      if (n < 0 || mark[n] || boxSet[n]) continue;
      mark[n] = 1;
      q.push(n);
    }
  }
  return { mark, min };
}

function boxSetOf(ctx: RoomCtx, boxes: number[]): Uint8Array {
  const s = new Uint8Array(ctx.C);
  for (const b of boxes) s[b] = 1;
  return s;
}

// True when the box configuration is certainly dead: a box off-goal on a
// dead cell or in a wall corner, or a frozen 2x2 block holding an off-goal box.
function boxesDead(ctx: RoomCtx, boxes: number[], set: Uint8Array): boolean {
  for (const b of boxes) {
    if (ctx.isGoal[b]) continue;
    if (ctx.deadCell[b] || ctx.cornerDead[b]) return true;
  }
  for (const blk of ctx.blocks) {
    let full = true;
    let bad = false;
    let any = false;
    for (const c of blk) {
      if (c < 0) continue;
      if (set[c]) {
        any = true;
        if (!ctx.isGoal[c]) bad = true;
      } else {
        full = false;
      }
    }
    if (full && any && bad) return true;
  }
  return false;
}

// Admissible heuristic: min-cost assignment of boxes to goals by single-box
// push distance. INF when no assignment exists (dead state).
function heuristic(ctx: RoomCtx, boxes: number[]): number {
  const C = ctx.C;
  let best = INF;
  for (const p of ctx.perms) {
    let s = 0;
    for (let j = 0; j < boxes.length; j++) {
      const d = ctx.dist[p[j] * C + boxes[j]];
      if (d >= INF) {
        s = INF;
        break;
      }
      s += d;
    }
    if (s < best) best = s;
  }
  return best;
}

function stateKey(ctx: RoomCtx, boxes: number[], player: number): number {
  const C = ctx.C;
  let k = player;
  // Base-C digits (player, b0, b1, b2, b3); missing boxes pad with C.
  for (let j = boxes.length - 1; j >= 0; j--) k = k * C + boxes[j];
  for (let j = boxes.length; j < 4; j++) k = k * C + C;
  return k;
}

// Binary min-heap of [f, h, seq, stateIndex], lexicographic order.
function heapPush(heap: number[][], e: number[]): void {
  heap.push(e);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (cmp(heap[p], heap[i]) <= 0) break;
    const t = heap[p];
    heap[p] = heap[i];
    heap[i] = t;
    i = p;
  }
}
function cmp(a: number[], b: number[]): number {
  if (a[0] !== b[0]) return a[0] - b[0];
  if (a[1] !== b[1]) return a[1] - b[1];
  return a[2] - b[2];
}
function heapPop(heap: number[][]): number[] {
  const top = heap[0];
  const last = heap.pop() as number[];
  if (heap.length > 0) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const l = i * 2 + 1;
      const r = l + 1;
      let m = i;
      if (l < heap.length && cmp(heap[l], heap[m]) < 0) m = l;
      if (r < heap.length && cmp(heap[r], heap[m]) < 0) m = r;
      if (m === i) break;
      const t = heap[m];
      heap[m] = heap[i];
      heap[i] = t;
      i = m;
    }
  }
  return top;
}

type Push = { box: number; dir: number };

// Push-based A* over one room. weight = 1 is optimal (consistent heuristic);
// larger weights trade optimality for speed. Returns the push list, or null
// when the node budget runs out (or the room is unsolvable).
function astarRoom(ctx: RoomCtx, weight: number, budget: number): { pushes: Push[]; expansions: number } | null {
  const startBoxes = ctx.boxes0.slice();
  const startSet = boxSetOf(ctx, startBoxes);
  const startPlayer = floodFill(ctx, ctx.entry, startSet).min;
  const h0 = heuristic(ctx, startBoxes);
  if (h0 >= INF) return null;
  const sBoxes: number[][] = [startBoxes];
  const sPlayer: number[] = [startPlayer];
  const sG: number[] = [0];
  const sH: number[] = [h0];
  const sParent: number[] = [-1];
  const sPushBox: number[] = [-1];
  const sPushDir: number[] = [-1];
  const best = new Map<number, number>();
  best.set(stateKey(ctx, startBoxes, startPlayer), 0);
  const heap: number[][] = [];
  let seq = 0;
  heapPush(heap, [h0, h0, seq++, 0]);
  let expansions = 0;
  while (heap.length > 0) {
    const top = heapPop(heap);
    const idx = top[3];
    const g = sG[idx];
    const key = stateKey(ctx, sBoxes[idx], sPlayer[idx]);
    if ((best.get(key) as number) < g) continue; // stale heap entry
    if (sH[idx] === 0) {
      // Reconstruct the push list.
      const pushes: Push[] = [];
      let cur = idx;
      while (sParent[cur] >= 0) {
        pushes.push({ box: sPushBox[cur], dir: sPushDir[cur] });
        cur = sParent[cur];
      }
      pushes.reverse();
      return { pushes, expansions };
    }
    if (++expansions > budget) return null;
    expansionsTotal++;
    const boxes = sBoxes[idx];
    const set = boxSetOf(ctx, boxes);
    const reach = floodFill(ctx, sPlayer[idx], set).mark;
    for (let j = 0; j < boxes.length; j++) {
      const b = boxes[j];
      for (let d = 0; d < 4; d++) {
        const stance = ctx.nbr[b * 4 + OPP[d]];
        const target = ctx.nbr[b * 4 + d];
        if (stance < 0 || target < 0) continue;
        if (!reach[stance] || set[target]) continue;
        const nb = boxes.slice();
        nb[j] = target;
        nb.sort((a, b2) => a - b2);
        const nset = boxSetOf(ctx, nb);
        if (boxesDead(ctx, nb, nset)) continue;
        const h = heuristic(ctx, nb);
        if (h >= INF) continue;
        const nplayer = floodFill(ctx, b, nset).min;
        const nkey = stateKey(ctx, nb, nplayer);
        const ng = g + 1;
        const prev = best.get(nkey);
        if (prev !== undefined && prev <= ng) continue;
        best.set(nkey, ng);
        const nidx = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(nplayer);
        sG.push(ng);
        sH.push(h);
        sParent.push(idx);
        sPushBox.push(b);
        sPushDir.push(d);
        heapPush(heap, [ng + weight * h, h, seq++, nidx]);
      }
    }
  }
  return null;
}

// Turn a push list into player moves, walking between pushes. `boxes` and
// `player` are the room's current (local) state at entry; they are updated.
function pushesToMoves(ctx: RoomCtx, pushes: Push[], boxes: number[], player: number): string {
  let moves = '';
  let curBoxes = boxes.slice();
  let curPlayer = player;
  const setOf = (bs: number[]): Uint8Array => boxSetOf(ctx, bs);
  for (const p of pushes) {
    const stance = ctx.nbr[p.box * 4 + OPP[p.dir]];
    const target = ctx.nbr[p.box * 4 + p.dir];
    // Walk from curPlayer to stance (boxes as obstacles).
    if (curPlayer !== stance) {
      const n = ctx.C;
      const prev = new Int32Array(n).fill(-1);
      const prevDir = new Int8Array(n).fill(-1);
      const bs = setOf(curBoxes);
      const q = [curPlayer];
      prev[curPlayer] = curPlayer;
      let head = 0;
      let found = false;
      while (head < q.length && !found) {
        const c = q[head++];
        for (let d = 0; d < 4; d++) {
          const ni = ctx.nbr[c * 4 + d];
          if (ni < 0 || bs[ni] || prev[ni] >= 0) continue;
          prev[ni] = c;
          prevDir[ni] = d;
          if (ni === stance) {
            found = true;
            break;
          }
          q.push(ni);
        }
      }
      if (!found) return moves; // should not happen; keep the valid prefix
      let s = '';
      let cur = stance;
      while (cur !== curPlayer) {
        s = DL[prevDir[cur]] + s;
        cur = prev[cur];
      }
      moves += s;
    }
    moves += DL[p.dir];
    curBoxes = curBoxes.slice();
    const j = curBoxes.indexOf(p.box);
    curBoxes[j] = target;
    curBoxes.sort((a, b) => a - b);
    curPlayer = p.box;
  }
  return moves;
}

// Solve one room. Tries optimal A*, then weighted, then greedy. Returns the
// move string and the room's final local player cell, or null on failure.
function solveRoom(ctx: RoomCtx): { moves: string; endPlayer: number; pushes: number } | null {
  const tries: Array<[number, number]> = [
    [1, BUDGET_OPT],
    [3, BUDGET_WEIGHTED],
    [1000, BUDGET_GREEDY],
  ];
  for (const [weight, budget] of tries) {
    const res = astarRoom(ctx, weight, budget);
    if (res) {
      const boxes = ctx.boxes0.slice();
      const moves = pushesToMoves(ctx, res.pushes, boxes, ctx.entry);
      // Final player cell: after the last push the player stands on the
      // pushed box's old cell.
      let endPlayer = ctx.entry;
      let curBoxes = ctx.boxes0.slice();
      for (const p of res.pushes) {
        const target = ctx.nbr[p.box * 4 + p.dir];
        const j = curBoxes.indexOf(p.box);
        curBoxes[j] = target;
        curBoxes.sort((a, b) => a - b);
        endPlayer = p.box;
      }
      if (DEBUG) {
        console.error(`room: weight=${weight} pushes=${res.pushes.length} expansions=${res.expansions} total=${expansionsTotal}`);
      }
      return { moves, endPlayer, pushes: res.pushes.length };
    }
    if (DEBUG) console.error(`room: weight=${weight} budget exhausted`);
  }
  return null;
}

// Solve one whole level: stitch the rooms in order (rooms whose door cell is
// a goal go last, so the player is never trapped inside by a box on the
// door). Returns the full move string.
function solveLevel(text: string): string {
  const L = parseLevel(text);
  if (L.player < 0) return '';
  const rooms = findRooms(L);
  // Rooms with a goal on the door cell last (a box ending there blocks exit).
  rooms.sort((a, b) => {
    const ga = L.goal[a.door] ? 1 : 0;
    const gb = L.goal[b.door] ? 1 : 0;
    if (ga !== gb) return ga - gb;
    return a.door - b.door;
  });
  const boxAt = new Uint8Array(L.w * L.h);
  boxAt.set(L.box);
  let player = L.player;
  let moves = '';
  // Local->global cell map, shared by all rooms of this level.
  const localToGlobal = new Map<number, number>();
  for (const room of rooms) {
    const walk = bfsPath(L, player, room.entry, boxAt);
    if (walk === null) return moves;
    moves += walk;
    player = room.entry;
    const ctx = buildRoomCtx(L, room);
    const solved = solveRoom(ctx);
    if (!solved) return moves;
    moves += solved.moves;
    // Update the global box map: room boxes are now on the room goals.
    for (const c of room.cells) boxAt[c] = 0;
    for (const c of room.cells) if (L.goal[c]) boxAt[c] = 1;
    // Map the room's final local player cell back to a global cell.
    const allowed = new Set<number>(room.cells);
    allowed.add(room.entry);
    const list = [...allowed].sort((a, b) => a - b);
    list.forEach((g, li) => localToGlobal.set(li, g));
    player = localToGlobal.get(solved.endPlayer) as number;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
