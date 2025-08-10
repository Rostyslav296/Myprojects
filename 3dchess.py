3D Chess in Python using pyglet + python-chess
Dependencies:
pip install pyglet python-chess
Run:
python 3d_chess.py
import math
import sys
import time

try:
import pyglet
from pyglet.gl import (
glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
glEnable, GL_DEPTH_TEST, glMatrixMode, GL_PROJECTION, glLoadIdentity,
gluPerspective, GL_MODELVIEW, gluLookAt, glViewport, glBegin, glEnd,
GL_QUADS, glColor4f, glColor3f, glVertex3f, glPushMatrix, glPopMatrix,
glTranslatef, glScalef, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
glDisable, GL_CULL_FACE, glRotatef, gluUnProject
)
except Exception:
print("This program requires pyglet. Install with: pip install pyglet")
sys.exit(1)

try:
import chess
except Exception:
print("This program requires python-chess. Install with: pip install python-chess")
sys.exit(1)

--------- Math helpers ---------
def deg2rad(d):
return d * math.pi / 180.0

def normalize(v):
l = math.sqrt(sum(c * c for c in v))
if l == 0:
return (0, 0, 0)
return (v[0] / l, v[1] / l, v[2] / l)

--------- Rendering helpers ---------
def draw_quad(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, color):
glColor4f(*color)
glBegin(GL_QUADS)
glVertex3f(x0, y0, z0)
glVertex3f(x1, y1, z1)
glVertex3f(x2, y2, z2)
glVertex3f(x3, y3, z3)
glEnd()

def draw_cube(center, size, color):
# Axis-aligned cube centered at 'center' with extents 'size' (sx, sy, sz)
cx, cy, cz = center
sx, sy, sz = size
x0, x1 = cx - sx / 2, cx + sx / 2
y0, y1 = cy - sy / 2, cy + sy / 2
z0, z1 = cz - sz / 2, cz + sz / 2
# 6 faces
# Front (z1)
draw_quad(x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1, color)
# Back (z0)
draw_quad(x1, y0, z0, x0, y0, z0, x0, y1, z0, x1, y1, z0, color)
# Left (x0)
draw_quad(x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0, color)
# Right (x1)
draw_quad(x1, y0, z1, x1, y0, z0, x1, y1, z0, x1, y1, z1, color)
# Top (y1)
draw_quad(x0, y1, z1, x1, y1, z1, x1, y1, z0, x0, y1, z0, color)
# Bottom (y0)
draw_quad(x0, y0, z0, x1, y0, z0, x1, y0, z1, x0, y0, z1, color)

--------- Chess to 3D mapping ---------
TILE_SIZE = 1.0
BOARD_Y = 0.0
BOARD_HALF = 4.0  # 8 * 0.5

def square_to_world(file_idx, rank_idx):
# file: 0..7 (a..h), rank: 0..7 (1..8)
x = (file_idx - 3.5) * TILE_SIZE
z = (rank_idx - 3.5) * TILE_SIZE
return x, BOARD_Y, z

def world_to_square(x, z):
filef = (x / TILE_SIZE) + 3.5
rankf = (z / TILE_SIZE) + 3.5
file_idx = int(math.floor(filef + 1e-6))
rank_idx = int(math.floor(rankf + 1e-6))
if 0 <= file_idx <= 7 and 0 <= rank_idx <= 7:
return file_idx, rank_idx
return None

def square_index(file_idx, rank_idx):
# python-chess square index: a1=0, ..., h8=63
return chess.square(file_idx, rank_idx)

def piece_color_rgba(piece):
# White pieces: light, Black pieces: dark
if piece.color == chess.WHITE:
return (0.9, 0.9, 0.95, 1.0)
else:
return (0.15, 0.15, 0.2, 1.0)

def piece_size_for_type(piece_type):
# Return (sx, sy, sz) sizes per piece type using cubes of various heights
base = 0.8
heights = {
chess.PAWN: 0.6,
chess.KNIGHT: 0.9,
chess.BISHOP: 1.0,
chess.ROOK: 1.1,
chess.QUEEN: 1.3,
chess.KING: 1.4,
}
h = heights.get(piece_type, 0.8)
return (base * 0.8, h, base * 0.8)

def draw_board():
# Alternating colors
light = (0.82, 0.74, 0.55, 1.0)
dark = (0.36, 0.25, 0.15, 1.0)
# Slight border
for file_idx in range(8):
for rank_idx in range(8):
x, y, z = square_to_world(file_idx, rank_idx)
c = light if (file_idx + rank_idx) % 2 == 0 else dark
sx, sz = TILE_SIZE, TILE_SIZE
# Draw tile as flat box slightly above y=0 to avoid z-fighting
draw_cube((x, y - 0.05, z), (sx, 0.1, sz), c)

def draw_highlight(file_idx, rank_idx, color=(0.2, 0.8, 0.2, 0.35)):
x, y, z = square_to_world(file_idx, rank_idx)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
draw_cube((x, y + 0.01, z), (TILE_SIZE * 0.98, 0.02, TILE_SIZE * 0.98), color)
glDisable(GL_BLEND)

def draw_pieces(board):
for sq in chess.SQUARES:
piece = board.piece_at(sq)
if not piece:
continue
f = chess.square_file(sq)
r = chess.square_rank(sq)
x, y, z = square_to_world(f, r)
sx, sy, sz = piece_size_for_type(piece.piece_type)
color = piece_color_rgba(piece)
draw_cube((x, y + sy / 2.0, z), (sx, sy, sz), color)

def square_under_mouse(window, proj_viewport):
mx, my = window._mouse_x, window._mouse_y
# Get two points in world space by unprojecting at depths 0 and 1
vx, vy, vw, vh = proj_viewport
# y must be window coords from bottom
win_x = mx
win_y = my
# Unproject
model = (pyglet.gl.GLdouble * 16)()
proj = (pyglet.gl.GLdouble * 16)()
viewport = (pyglet.gl.GLint * 4)(int(vx), int(vy), int(vw), int(vh))
pyglet.gl.glGetDoublev(pyglet.gl.GL_MODELVIEW_MATRIX, model)
pyglet.gl.glGetDoublev(pyglet.gl.GL_PROJECTION_MATRIX, proj)

near_x = pyglet.gl.GLdouble()
near_y = pyglet.gl.GLdouble()
near_z = pyglet.gl.GLdouble()
far_x = pyglet.gl.GLdouble()
far_y = pyglet.gl.GLdouble()
far_z = pyglet.gl.GLdouble()

if not gluUnProject(win_x, win_y, 0.0, model, proj, viewport, near_x, near_y, near_z):
    return None
if not gluUnProject(win_x, win_y, 1.0, model, proj, viewport, far_x, far_y, far_z):
    return None

r0 = (near_x.value, near_y.value, near_z.value)
r1 = (far_x.value, far_y.value, far_z.value)
rd = (r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2])

if abs(rd[1]) < 1e-6:
    return None
t = (BOARD_Y - r0[1]) / rd[1]
if t < 0:
    return None
hit = (r0[0] + rd[0] * t, r0[1] + rd[1] * t, r0[2] + rd[2] * t)
sq = world_to_square(hit[0], hit[2])
return sq
--------- Main App ---------
class Chess3DWindow(pyglet.window.Window):
def init(self, width=1000, height=700):
super().init(width=width, height=height, caption="3D Chess (pyglet + python-chess)", resizable=True)
glClearColor(0.08, 0.12, 0.16, 1.0)
glEnable(GL_DEPTH_TEST)
glDisable(GL_CULL_FACE)
self.fps_display = pyglet.window.FPSDisplay(self)
self.board = chess.Board()
self.selected = None  # (file, rank)
self.legal_targets = set()
self.status_label = pyglet.text.Label(
"", font_size=14, x=10, y=self.height - 20, color=(255, 255, 255, 255)
)
self.turn_label = pyglet.text.Label(
"", font_size=14, x=10, y=self.height - 40, color=(200, 220, 255, 255)
)
self.help_label = pyglet.text.Label(
"LMB: select/move  RMB+Drag: orbit  Scroll: zoom  R: reset  Q: quit",
font_size=12, x=10, y=10, color=(180, 200, 220, 255)
)
# Camera params
self.distance = 11.0
self.azimuth = 45.0    # around Y
self.elevation = 35.0  # above XZ plane
self.center = (0.0, 0.0, 0.0)
self._right_drag = False
self._last_mouse = (0, 0)
self._projection = (0, 0, width, height)  # viewport for unproject
self._mouse_x = 0
self._mouse_y = 0
self.last_move_time = time.time()

    pyglet.clock.schedule_interval(self.update_labels, 0.2)

def update_labels(self, dt=0):
    if self.board.is_checkmate():
        msg = "Checkmate! " + ("Black" if self.board.turn == chess.WHITE else "White") + " wins. Press R to reset."
    elif self.board.is_stalemate():
        msg = "Stalemate. Press R to reset."
    elif self.board.is_insufficient_material():
        msg = "Draw by insufficient material. Press R to reset."
    elif self.board.is_check():
        msg = "Check!"
    else:
        msg = ""
    self.status_label.text = msg
    self.turn_label.text = f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}"

def on_resize(self, width, height):
    super().on_resize(width, height)
    glViewport(0, 0, width, height)
    self._projection = (0, 0, width, height)

def apply_camera(self):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = max(self.width / float(self.height), 1e-3)
    gluPerspective(45.0, aspect, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Spherical camera
    elev = deg2rad(self.elevation)
    azim = deg2rad(self.azimuth)
    cx = self.center[0]
    cy = self.center[1]
    cz = self.center[2]
    cam_x = cx + self.distance * math.cos(elev) * math.sin(azim)
    cam_y = cy + self.distance * math.sin(elev)
    cam_z = cz + self.distance * math.cos(elev) * math.cos(azim)
    # Look a bit towards white's side (negative z) to align perspective
    gluLookAt(cam_x, cam_y, cam_z, cx, cy, cz, 0, 1, 0)

def on_draw(self):
    self.clear()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    self.apply_camera()

    # Ground plane under board
    draw_cube((0, -0.26, 0), (10.0, 0.5, 10.0), (0.10, 0.12, 0.14, 1.0))

    draw_board()
    if self.selected:
        draw_highlight(*self.selected, color=(0.8, 0.8, 0.2, 0.35))
        for t in self.legal_targets:
            draw_highlight(*t, color=(0.2, 0.8, 0.2, 0.35))
    draw_pieces(self.board)

    # 2D overlays
    self.status_label.y = self.height - 20
    self.turn_label.y = self.height - 40
    self.status_label.draw()
    self.turn_label.draw()
    self.help_label.draw()
    self.fps_display.draw()

def get_square_under_mouse(self):
    return square_under_mouse(self, self._projection)

def on_mouse_motion(self, x, y, dx, dy):
    self._mouse_x = x
    self._mouse_y = y

def on_mouse_press(self, x, y, button, modifiers):
    self._mouse_x = x
    self._mouse_y = y
    if button == pyglet.window.mouse.RIGHT:
        self._right_drag = True
        self._last_mouse = (x, y)
        return

    if button == pyglet.window.mouse.LEFT:
        if self.board.is_game_over():
            return
        sq = self.get_square_under_mouse()
        if sq is None:
            self.selected = None
            self.legal_targets = set()
            return
        f, r = sq
        s_idx = square_index(f, r)
        piece = self.board.piece_at(s_idx)
        if self.selected is None:
            # Select if there's a piece of current side
            if piece and piece.color == self.board.turn:
                self.selected = (f, r)
                self.legal_targets = self._legal_targets_from_square(s_idx)
            else:
                self.selected = None
                self.legal_targets = set()
        else:
            # Attempt move from selected to clicked
            from_f, from_r = self.selected
            to_f, to_r = f, r
            self.try_move(from_f, from_r, to_f, to_r)

def on_mouse_release(self, x, y, button, modifiers):
    if button == pyglet.window.mouse.RIGHT:
        self._right_drag = False

def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
    self._mouse_x = x
    self._mouse_y = y
    if self._right_drag:
        self.azimuth = (self.azimuth + dx * 0.4) % 360.0
        self.elevation = max(10.0, min(85.0, self.elevation + dy * 0.2))

def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
    self.distance = max(6.0, min(24.0, self.distance - scroll_y * 0.7))

def on_key_press(self, symbol, modifiers):
    if symbol in (pyglet.window.key.Q, pyglet.window.key.ESCAPE):
        self.close()
    if symbol == pyglet.window.key.R:
        self.board.reset()
        self.selected = None
        self.legal_targets = set()

def _legal_targets_from_square(self, from_sq_idx):
    targets = set()
    for mv in self.board.legal_moves:
        if mv.from_square == from_sq_idx:
            tf = chess.square_file(mv.to_square)
            tr = chess.square_rank(mv.to_square)
            targets.add((tf, tr))
    return targets

def try_move(self, from_f, from_r, to_f, to_r):
    from_sq = square_index(from_f, from_r)
    to_sq = square_index(to_f, to_r)

    # Handle promotions automatically to Queen unless another is forced
    promo = None
    piece = self.board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and to_r == 7) or (piece.color == chess.BLACK and to_r == 0):
            promo = chess.QUEEN

    candidate = chess.Move(from_sq, to_sq, promotion=promo)
    if candidate in self.board.legal_moves:
        self.board.push(candidate)
        self.selected = None
        self.legal_targets = set()
        self.last_move_time = time.time()
        return

    # Maybe non-promotion move if promo not needed
    candidate = chess.Move(from_sq, to_sq)
    if candidate in self.board.legal_moves:
        self.board.push(candidate)
        self.selected = None
        self.legal_targets = set()
        self.last_move_time = time.time()
        return

    # If invalid, try re-selecting
    piece = self.board.piece_at(square_index(to_f, to_r))
    if piece and piece.color == self.board.turn:
        self.selected = (to_f, to_r)
        self.legal_targets = self._legal_targets_from_square(square_index(to_f, to_r))
    else:
        self.selected = None
        self.legal_targets = set()
if name == "main":
window = Chess3DWindow(1000, 720)
pyglet.app.run()