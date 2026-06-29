import HAL
import WebGUI
import Frequency

import math
import random
import time
import numpy as np


# ============================================================
# MONTE CARLO LASER LOC - ROBOTICS ACADEMY
# Atividade AB2 Robótica Móvel
# Autoria: Rita Lemos (rklp@ic.ufal.br)
#
# Implementa:
# 1. Inicialização uniforme das partículas em regiões livres
# 2. Propagação com odometria ruidosa
# 3. Modelo de observação por beam model / laser virtual
# 4. Atualização de pesos com laser real x laser simulado
# 5. Resampling sistemático
# 6. Estimação final por melhor cluster + odometria ruidosa
#
# HAL.getPose3d() é usado somente para debug no terminal.
# O algoritmo usa HAL.getOdom() + HAL.getLaserData().
# ============================================================


# ============================================================
# PARÂMETROS
# ============================================================

N_PARTICLES = 300
LASER_NUM_BEAMS = 24

MAP_URL = "/resources/exercises/montecarlo_laser_loc/images/mapgrannyannie.png"

LASER_MAX = 10.0

# Ray casting em pixels
RAY_STEP_PX = 4.0

WEIGHT_TEMPERATURE = 0.0030
UNIFORM_MIX = 0.004

USE_ODOM_PRIOR = True

ODOM_PRIOR_GAIN = 2.20
ODOM_PRIOR_SIGMA_XY = 0.45
ODOM_PRIOR_SIGMA_YAW = 0.45

ODOM_INJECTION_RATE_NORMAL = 0.42
GLOBAL_RANDOM_RATE_NORMAL = 0.002

# Ruído do motion model
ODOM_TRANS_NOISE_BASE = 0.006
ODOM_TRANS_NOISE_GAIN = 0.040
ODOM_ROT_NOISE_BASE = 0.006
ODOM_ROT_NOISE_GAIN = 0.045

# O filtro só atualiza depois de movimento mínimo
MIN_TRANS_UPDATE = 0.04
MIN_ROT_UPDATE = 0.06
MAX_TIME_UPDATE = 0.60

# Resampling
RESAMPLE_NEFF_RATIO = 0.60

# Reinjeção de partículas
GLOBAL_RANDOM_RATE_INITIAL = 0.06

ODOM_INJECTION_RATE_INITIAL = 0.28

# Fase inicial: gira parado para coletar observações
INIT_SPIN_UPDATES = 25

# Loop
LOOP_HZ = 20
PRINT_EVERY = 5

# Visualização
SHOW_PARTICLES = True
SHOW_ESTIMATED_ROBOT = True

DEBUG_REAL_POSE = True

# Movimento automático
AUTO_MOVE = True


# ============================================================
# GLOBAIS
# ============================================================

particles = None

last_filter_odom = None
last_filter_time = time.time()

sensor_updates = 0
t0 = time.time()


# ============================================================
# AUXILIARES
# ============================================================

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def safe(v):
    try:
        v = float(v)
    except Exception:
        return LASER_MAX

    if math.isnan(v) or math.isinf(v):
        return LASER_MAX

    return max(0.02, min(v, LASER_MAX))


def gaussian(sigma):
    return random.gauss(0.0, sigma)


def blend_angle(a, b, beta):
    """
    Mistura circular entre dois ângulos.
    beta = quanto do segundo ângulo entra.
    """
    s = (1.0 - beta) * math.sin(a) + beta * math.sin(b)
    c = (1.0 - beta) * math.cos(a) + beta * math.cos(b)
    return math.atan2(s, c)


# ============================================================
# MAPA
# ============================================================

raw = np.array(WebGUI.getMap(MAP_URL)).astype(float)

if raw.max() > 1.5:
    raw /= 255.0

gray = np.mean(raw[:, :, :3], axis=2)

MAP_H, MAP_W = gray.shape

# No mapa: branco = livre, preto = parede/obstáculo
occupied = gray < 0.50
free = np.logical_not(occupied)

free_pixels = np.argwhere(free)  # [my, mx]


# ============================================================
# TRANSFORMAÇÃO MUNDO <-> MAPA
# ============================================================

_mx0, _my0, _ = WebGUI.poseToMap(0.0, 0.0, 0.0)
_mx1, _my1, _ = WebGUI.poseToMap(1.0, 0.0, 0.0)
_mx2, _my2, _ = WebGUI.poseToMap(0.0, 1.0, 0.0)

OX = float(_mx0)
OY = float(_my0)

XX = float(_mx1 - _mx0)
XY = float(_my1 - _my0)
YX = float(_mx2 - _mx0)
YY = float(_my2 - _my0)

A = np.array([
    [XX, YX],
    [XY, YY]
], dtype=float)

A_INV = np.linalg.inv(A)

SCALE_X = math.sqrt(XX * XX + XY * XY)
SCALE_Y = math.sqrt(YX * YX + YY * YY)
SCALE = 0.5 * (SCALE_X + SCALE_Y)


def w2m(x, y):
    mx = OX + XX * x + YX * y
    my = OY + XY * x + YY * y
    return int(round(mx)), int(round(my))


def m2w(mx, my):
    p = np.array([float(mx) - OX, float(my) - OY])
    w = A_INV @ p
    return float(w[0]), float(w[1])


def inside(mx, my):
    return 0 <= mx < MAP_W and 0 <= my < MAP_H


def free_pixel(mx, my, margin=2):
    mx = int(mx)
    my = int(my)

    if not inside(mx, my):
        return False

    x0 = max(0, mx - margin)
    x1 = min(MAP_W, mx + margin + 1)

    y0 = max(0, my - margin)
    y1 = min(MAP_H, my + margin + 1)

    return not np.any(occupied[y0:y1, x0:x1])


def free_pose(x, y):
    mx, my = w2m(x, y)
    return free_pixel(mx, my, margin=2)


# ============================================================
# INICIALIZAÇÃO DE PARTÍCULAS
# ============================================================

def random_free_particle():
    for _ in range(3000):
        idx = random.randint(0, len(free_pixels) - 1)
        my, mx = free_pixels[idx]

        if free_pixel(mx, my, margin=3):
            x, y = m2w(mx, my)
            yaw = random.uniform(-math.pi, math.pi)
            return [x, y, yaw, 1.0 / N_PARTICLES]

    return [
        random.uniform(-4.0, 4.0),
        random.uniform(-4.0, 4.0),
        random.uniform(-math.pi, math.pi),
        1.0 / N_PARTICLES
    ]


def random_odom_particle():
    """
    Partícula próxima da odometria ruidosa.

    """
    odom = HAL.getOdom()

    if sensor_updates < INIT_SPIN_UPDATES:
        sigma_xy = 0.85
        sigma_yaw = 1.10
    else:
        sigma_xy = 0.22
        sigma_yaw = 0.28

    for _ in range(200):
        x = odom.x + gaussian(sigma_xy)
        y = odom.y + gaussian(sigma_xy)
        yaw = wrap(odom.yaw + gaussian(sigma_yaw))

        if free_pose(x, y):
            return [x, y, yaw, 1.0 / N_PARTICLES]

    return random_free_particle()


def init_particles():
    global particles

    particles = np.zeros((N_PARTICLES, 4), dtype=float)

    # Inicialização uniforme no mapa livre
    for i in range(N_PARTICLES):
        particles[i] = random_free_particle()

    particles[:, 3] = 1.0 / N_PARTICLES


# ============================================================
# LASER
# ============================================================

# PDF:
# índice 90 = frente
# índice 0 = direita
# índice 179 = esquerda
# angle = radians(i - 90)
BEAM_IDX = np.linspace(8, 171, LASER_NUM_BEAMS).astype(int)

BEAM_ANG = np.array(
    [math.radians(int(i) - 90) for i in BEAM_IDX],
    dtype=float
)


def get_real_laser(laser_data):
    values = laser_data.values
    return np.array([safe(values[int(i)]) for i in BEAM_IDX], dtype=float)


# ============================================================
# BEAM MODEL / LASER VIRTUAL
# ============================================================

def ray_cast_particle(x, y, yaw, beam_angle):
    """
    Simula uma leitura de laser saindo da partícula.
    """
    mx0, my0 = w2m(x, y)

    if not inside(mx0, my0):
        return LASER_MAX

    # Ponto auxiliar 1 m à frente na direção do feixe
    wx1 = x + math.cos(yaw + beam_angle)
    wy1 = y + math.sin(yaw + beam_angle)

    mx1, my1 = w2m(wx1, wy1)

    dx = mx1 - mx0
    dy = my1 - my0

    norm = math.sqrt(dx * dx + dy * dy)

    if norm < 1e-6:
        return LASER_MAX

    ux = dx / norm
    uy = dy / norm

    max_px = LASER_MAX * SCALE
    d = RAY_STEP_PX

    while d < max_px:
        mx = int(round(mx0 + ux * d))
        my = int(round(my0 + uy * d))

        if not inside(mx, my):
            return d / SCALE

        if occupied[my, mx]:
            return d / SCALE

        d += RAY_STEP_PX

    return LASER_MAX


def simulated_laser(pose):
    x, y, yaw = pose

    readings = []

    for a in BEAM_ANG:
        readings.append(ray_cast_particle(x, y, yaw, a))

    return np.array(readings, dtype=float)


def particle_error(pose, real_ranges):
    """
    MSE normalizado entre laser real e laser virtual.
    """
    x, y, yaw = pose

    if not free_pose(x, y):
        return 999.0

    sim_ranges = simulated_laser(pose)

    real_norm = real_ranges / LASER_MAX
    sim_norm = sim_ranges / LASER_MAX

    # Leituras no alcance máximo informam pouco
    valid = real_ranges < LASER_MAX * 0.98

    if np.sum(valid) < 4:
        valid = np.ones_like(real_ranges, dtype=bool)

    mse = np.mean((real_norm[valid] - sim_norm[valid]) ** 2)

    return float(mse)


# ============================================================
# ATUALIZAÇÃO DOS PESOS
# ============================================================

def update_weights(laser_data):
    global particles

    real_ranges = get_real_laser(laser_data)

    errors = np.zeros(N_PARTICLES, dtype=float)

    for i in range(N_PARTICLES):
        errors[i] = particle_error(particles[i, :3], real_ranges)

    # Peso relativo ao melhor erro
    min_error = float(np.min(errors))
    relative_errors = errors - min_error

    log_w = -relative_errors / WEIGHT_TEMPERATURE

    # Prior fraco pela odometria absoluta ruidosa
    if USE_ODOM_PRIOR:
        odom = HAL.getOdom()

        dx = particles[:, 0] - odom.x
        dy = particles[:, 1] - odom.y

        dist2 = dx * dx + dy * dy

        yaw_diff = np.array([
            wrap(float(particles[i, 2] - odom.yaw))
            for i in range(N_PARTICLES)
        ])

        log_odom = (
            -0.5 * dist2 / (ODOM_PRIOR_SIGMA_XY * ODOM_PRIOR_SIGMA_XY)
            -0.5 * yaw_diff * yaw_diff / (ODOM_PRIOR_SIGMA_YAW * ODOM_PRIOR_SIGMA_YAW)
        )

        log_w = log_w + ODOM_PRIOR_GAIN * log_odom

    max_log = np.max(log_w)

    if not np.isfinite(max_log):
        particles[:, 3] = 1.0 / N_PARTICLES
        return particles[:, 3], errors

    w = np.exp(log_w - max_log)

    total = np.sum(w)

    if total <= 0.0 or not np.isfinite(total):
        particles[:, 3] = 1.0 / N_PARTICLES
        return particles[:, 3], errors

    w = w / total

    # Mistura uniforme pequena
    w = (1.0 - UNIFORM_MIX) * w + UNIFORM_MIX * (np.ones(N_PARTICLES) / N_PARTICLES)
    w = w / np.sum(w)

    particles[:, 3] = w

    return particles[:, 3], errors


# ============================================================
# MOTION MODEL COM ODOMETRIA
# ============================================================

def odom_delta(prev, curr):
    dx = curr.x - prev.x
    dy = curr.y - prev.y
    dyaw = wrap(curr.yaw - prev.yaw)

    trans = math.sqrt(dx * dx + dy * dy)

    return trans, abs(dyaw)


def should_run_filter_step(prev, curr):
    global last_filter_time

    trans, rot = odom_delta(prev, curr)
    elapsed = time.time() - last_filter_time

    return (
        trans >= MIN_TRANS_UPDATE or
        rot >= MIN_ROT_UPDATE or
        elapsed >= MAX_TIME_UPDATE
    )


def propagate_particles_from_odom(prev_odom, curr_odom):
    """
    Propaga usando a odometria acumulada desde a última atualização.
    """
    global particles

    dx_global = curr_odom.x - prev_odom.x
    dy_global = curr_odom.y - prev_odom.y
    dyaw = wrap(curr_odom.yaw - prev_odom.yaw)

    # Deslocamento global -> deslocamento local no referencial anterior
    c = math.cos(prev_odom.yaw)
    s = math.sin(prev_odom.yaw)

    dx_local = c * dx_global + s * dy_global
    dy_local = -s * dx_global + c * dy_global

    trans = math.sqrt(dx_local * dx_local + dy_local * dy_local)

    sigma_xy = ODOM_TRANS_NOISE_BASE + ODOM_TRANS_NOISE_GAIN * abs(trans)
    sigma_yaw = ODOM_ROT_NOISE_BASE + ODOM_ROT_NOISE_GAIN * abs(dyaw)

    for i in range(N_PARTICLES):
        x, y, yaw = particles[i, :3]

        ndx = dx_local + gaussian(sigma_xy)
        ndy = dy_local + gaussian(sigma_xy)
        ndyaw = dyaw + gaussian(sigma_yaw)

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        nx = x + ndx * cy - ndy * sy
        ny = y + ndx * sy + ndy * cy
        nyaw = wrap(yaw + ndyaw)

        if free_pose(nx, ny):
            particles[i, 0] = nx
            particles[i, 1] = ny
            particles[i, 2] = nyaw
        else:
            particles[i] = random_free_particle()


# ============================================================
# RESAMPLING
# ============================================================

def effective_sample_size(weights):
    return 1.0 / np.sum(weights * weights)


def current_rates():
    if sensor_updates < INIT_SPIN_UPDATES:
        return GLOBAL_RANDOM_RATE_INITIAL, ODOM_INJECTION_RATE_INITIAL

    return GLOBAL_RANDOM_RATE_NORMAL, ODOM_INJECTION_RATE_NORMAL


def resample_systematic(weights):
    global particles

    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0

    step = 1.0 / N_PARTICLES
    start = random.uniform(0.0, step)

    positions = start + step * np.arange(N_PARTICLES)
    indexes = np.searchsorted(cumulative, positions)

    new_particles = particles[indexes].copy()

    # Pequeno ruído após resampling
    for i in range(N_PARTICLES):
        new_particles[i, 0] += gaussian(0.010)
        new_particles[i, 1] += gaussian(0.010)
        new_particles[i, 2] = wrap(new_particles[i, 2] + gaussian(0.018))

        if not free_pose(new_particles[i, 0], new_particles[i, 1]):
            new_particles[i] = random_free_particle()

    global_rate, odom_rate = current_rates()

    # Reinjeção global
    n_global = int(global_rate * N_PARTICLES)

    for _ in range(n_global):
        j = random.randint(0, N_PARTICLES - 1)
        new_particles[j] = random_free_particle()

    # Reinjeção próxima da odometria
    n_odom = int(odom_rate * N_PARTICLES)

    for _ in range(n_odom):
        j = random.randint(0, N_PARTICLES - 1)
        new_particles[j] = random_odom_particle()

    new_particles[:, 3] = 1.0 / N_PARTICLES

    particles = new_particles


# ============================================================
# ESTIMAÇÃO
# ============================================================

def estimate_pose():
    w = particles[:, 3]

    best_idx = int(np.argmax(w))

    bx = particles[best_idx, 0]
    by = particles[best_idx, 1]

    d2 = (particles[:, 0] - bx) ** 2 + (particles[:, 1] - by) ** 2

    mask = d2 < 0.45 * 0.45

    if np.sum(mask) < 8:
        k = max(10, int(0.10 * N_PARTICLES))
        idx = np.argsort(w)[-k:]
    else:
        idx = np.where(mask)[0]

    selected = particles[idx]
    sw = selected[:, 3].copy()

    total = np.sum(sw)

    if total <= 0.0 or not np.isfinite(total):
        sw[:] = 1.0 / len(sw)
    else:
        sw /= total

    x_pf = float(np.sum(sw * selected[:, 0]))
    y_pf = float(np.sum(sw * selected[:, 1]))

    yaw_pf = math.atan2(
        float(np.sum(sw * np.sin(selected[:, 2]))),
        float(np.sum(sw * np.cos(selected[:, 2])))
    )

    odom = HAL.getOdom()

    dist_odom = math.sqrt((x_pf - odom.x) ** 2 + (y_pf - odom.y) ** 2)

    if sensor_updates < INIT_SPIN_UPDATES:
        beta = 0.15
    elif dist_odom > 1.00:
        beta = 0.80
    elif dist_odom > 0.50:
        beta = 0.65
    else:
        beta = 0.45
    

    x_est = (1.0 - beta) * x_pf + beta * odom.x
    y_est = (1.0 - beta) * y_pf + beta * odom.y
    yaw_est = blend_angle(yaw_pf, odom.yaw, beta)

    return x_est, y_est, yaw_est


# ============================================================
# VISUALIZAÇÃO
# ============================================================

def show_particles():
    if not SHOW_PARTICLES:
        return

    w = particles[:, 3]
    wmax = np.max(w) + 1e-12

    msg = []

    for i in range(N_PARTICLES):
        visual_weight = 0.03 + 0.25 * math.sqrt(float(w[i] / wmax))
        visual_weight = max(0.03, min(0.35, visual_weight))

        msg.append([
            float(particles[i, 0]),
            float(particles[i, 1]),
            float(particles[i, 2]),
            visual_weight
        ])

    WebGUI.showParticles(msg)


def show_estimated_pose(x, y, yaw):
    if SHOW_ESTIMATED_ROBOT:
        WebGUI.showPosition(x, y, yaw)


# ============================================================
# MOVIMENTO
# ============================================================

def move_robot(laser_data):
    if not AUTO_MOVE:
        HAL.setV(0.0)
        HAL.setW(0.0)
        return

    values = laser_data.values

    if len(values) < 180:
        HAL.setV(0.0)
        HAL.setW(0.0)
        return

    # Fase inicial: gira parado
    if sensor_updates < INIT_SPIN_UPDATES:
        HAL.setV(0.0)
        HAL.setW(0.45)
        return

    front = min(safe(v) for v in values[75:105])
    left = min(safe(v) for v in values[110:170])
    right = min(safe(v) for v in values[10:70])

    if front < 0.55:
        HAL.setV(-0.04)
        HAL.setW(0.85 if left > right else -0.85)

    elif front < 0.90:
        HAL.setV(0.08)
        HAL.setW(0.45 if left > right else -0.45)

    else:
        HAL.setV(0.12)

        if left < 0.55:
            HAL.setW(-0.30)
        elif right < 0.55:
            HAL.setW(0.30)
        else:
            HAL.setW(0.10)


# ============================================================
# PRINT
# ============================================================

def print_status(weights, errors, x_est, y_est, yaw_est):
    elapsed = time.time() - t0
    neff = effective_sample_size(weights)

    phase = "SPIN" if sensor_updates < INIT_SPIN_UPDATES else "MOVING"

    if DEBUG_REAL_POSE:
        real = HAL.getPose3d()
        odom = HAL.getOdom()

        print(
            "t={:.1f} {} upd={} Neff={:.0f} ErrMin={:.5f} ErrMed={:.5f} Est=({:.2f},{:.2f},{:.2f}) Odom=({:.2f},{:.2f},{:.2f}) Real=({:.2f},{:.2f},{:.2f})".format(
                elapsed,
                phase,
                sensor_updates,
                neff,
                float(np.min(errors)),
                float(np.median(errors)),
                x_est,
                y_est,
                yaw_est,
                odom.x,
                odom.y,
                odom.yaw,
                real.x,
                real.y,
                real.yaw
            ),
            flush=True
        )

    else:
        print(
            "t={:.1f} {} upd={} Neff={:.0f} ErrMin={:.5f} ErrMed={:.5f} Est=({:.2f},{:.2f},{:.2f})".format(
                elapsed,
                phase,
                sensor_updates,
                neff,
                float(np.min(errors)),
                float(np.median(errors)),
                x_est,
                y_est,
                yaw_est
            ),
            flush=True
        )


# ============================================================
# MAIN
# ============================================================

init_particles()

print("Monte Carlo Laser Loc iniciado.", flush=True)
print("N_PARTICLES =", N_PARTICLES, flush=True)
print("LASER_NUM_BEAMS =", LASER_NUM_BEAMS, flush=True)
print("WEIGHT_TEMPERATURE =", WEIGHT_TEMPERATURE, flush=True)
print("ODOM_PRIOR_GAIN =", ODOM_PRIOR_GAIN, flush=True)
print("ODOM_INJECTION_RATE_NORMAL =", ODOM_INJECTION_RATE_NORMAL, flush=True)
print("MAP_SCALE = {:.2f} px/m".format(SCALE), flush=True)
print("Filtro: move -> odom/laser -> motion -> pesos -> resampling.", flush=True)

while True:
    laser = HAL.getLaserData()
    curr_odom = HAL.getOdom()

    if last_filter_odom is None:
        last_filter_odom = curr_odom
        last_filter_time = time.time()

    # Executa uma etapa do filtro após movimento mínimo
    if should_run_filter_step(last_filter_odom, curr_odom):

        # 1. Motion model
        propagate_particles_from_odom(last_filter_odom, curr_odom)

        # 2. Sensor model
        weights, errors = update_weights(laser)

        # 3. Estimate
        x_est, y_est, yaw_est = estimate_pose()

        # 4. Visual
        show_estimated_pose(x_est, y_est, yaw_est)
        show_particles()

        # 5. Debug
        if sensor_updates % PRINT_EVERY == 0:
            print_status(weights, errors, x_est, y_est, yaw_est)

        # 6. Resampling
        neff = effective_sample_size(weights)

        if neff < RESAMPLE_NEFF_RATIO * N_PARTICLES:
            resample_systematic(weights)

        last_filter_odom = curr_odom
        last_filter_time = time.time()
        sensor_updates += 1

    # 7. Move robot
    move_robot(laser)

    Frequency.tick(LOOP_HZ)