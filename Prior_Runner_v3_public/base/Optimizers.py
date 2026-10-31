# Optimizers.py
import numpy as np
import random
from Region import Region, clip_region

#********************************

# ====================== UTILIDADES COMUNES (GLOBAL) ======================
def _to_region(pos, img_shape):
    """Convierte array [x,y,w,h] a objeto Region válido."""
    h, w = img_shape
    x, y, rw, rh = np.round(pos).astype(int)
    x = max(0, min(x, w - 50))
    y = max(0, min(y, h - 50))
    rw = max(50, min(rw, w - x))
    rh = max(50, min(rh, h - y))
    return Region(x, y, rw, rh)

# --- 1. Genetic Algorithm (GA) - Usado como Base ---

def tournament_select(pop, gray, fitness_func, k=3):
    """Selección por Torneo."""
    candidates = random.sample(pop, min(k, len(pop)))
    best = max(candidates, key=lambda r: fitness_func(r, gray))
    return best

def crossover(p1, p2):
    """Cruce de un punto de los atributos de la Región."""
    c1 = Region(p1.x, p1.y, p1.w, p1.h)
    c2 = Region(p2.x, p2.y, p2.w, p2.h)
    for attr in ['x', 'y', 'w', 'h']:
        if random.random() < 0.5:
            setattr(c1, attr, getattr(p2, attr))
            setattr(c2, attr, getattr(p1, attr))
    return c1, c2

def mutate(reg, img_shape, pmut=0.01):
    """Mutación aleatoria en la posición/tamaño de la Región."""
    h, w = img_shape
    for attr in ['x', 'y']:
        if random.random() < pmut:
            setattr(reg, attr, getattr(reg, attr) + random.randint(-10, 10))
    for attr in ['w', 'h']:
        if random.random() < pmut:
            setattr(reg, attr, getattr(reg, attr) + random.randint(-5, 5))
    clip_region(reg, (h, w))
    return reg

def ga_optimize(leaves, gray, fitness_func, pop_size=20, gens=30, pcross=0.8, pmut=0.01):
    """Implementación del Algoritmo Genético (GA)."""
    h, w = gray.shape
    # Inicialización de la población con regiones del QuadTree
    pop = random.sample(leaves, min(pop_size, len(leaves)))
    pop = [Region(leaf.x, leaf.y, leaf.w, leaf.h) for leaf in pop]
    
    for _ in range(gens):
        new_pop = []
        for _ in range(pop_size // 2):
            parent1 = tournament_select(pop, gray, fitness_func)
            parent2 = tournament_select(pop, gray, fitness_func)
            
            if random.random() < pcross:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1 = Region(parent1.x, parent1.y, parent1.w, parent1.h)
                child2 = Region(parent2.x, parent2.y, parent2.w, parent2.h)
                
            child1 = mutate(child1, (h, w), pmut)
            child2 = mutate(child2, (h, w), pmut)
            new_pop += [child1, child2]
        pop = new_pop

    best = max(pop, key=lambda r: fitness_func(r, gray))
    return best

# --- 2. Interfaces para Otros Optimizadores ---

# NOTA: Los siguientes optimizadores requieren una implementación completa.
# Deben simular la búsqueda de un conjunto de parámetros (x, y, w, h) que maximicen la función de fitness.

# ====================== 2. PSO - Particle Swarm Optimization ======================
def pso_optimize(leaves, gray, fitness_func, pop_size=30, gens=40):
    h, w = gray.shape
    dim = 4  # x, y, w, h
    w_inertia = 0.9
    c1, c2 = 2.0, 2.0

    # Inicialización
    X = np.array([[random.randint(0, w//3), random.randint(0, h//3),
                   random.randint(w//4, w//2), random.randint(h//4, h//2)] for _ in range(pop_size)])
    V = np.zeros_like(X, dtype=float)
    pbest = X.copy()
    pbest_fit = np.array([fitness_func(Region(*pos), gray) for pos in X])
    gbest_idx = np.argmax(pbest_fit)
    gbest = X[gbest_idx]
    gbest_fit = pbest_fit[gbest_idx]

    for _ in range(gens):
        r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
        V = w_inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V.astype(int)
        X[:, 0] = np.clip(X[:, 0], 0, w-50)
        X[:, 1] = np.clip(X[:, 1], 0, h-50)
        X[:, 2] = np.clip(X[:, 2], 50, w - X[:, 0])
        X[:, 3] = np.clip(X[:, 3], 50, h - X[:, 1])

        for i in range(pop_size):
            reg = Region(*X[i])
            fit = fitness_func(reg, gray)
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = X[i]
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest = X[i]

        w_inertia = 0.9 - 0.5 * (_ / gens)  # decaimiento lineal

    return Region(*gbest.astype(int))


# ====================== 3. CSA - Crow Search Algorithm (reemplaza ACO) ======================
def cso_optimize(leaves, gray, fitness_func, pop_size=30, gens=40, AP=0.1, fl=2.0):
    h, w = gray.shape
    crows = np.array([[random.randint(0, w//3), random.randint(0, h//3),
                       random.randint(w//4, w//2), random.randint(h//4, h//2)] for _ in range(pop_size)])
    memory = crows.copy()
    fitness = np.array([fitness_func(Region(*pos), gray) for pos in crows])
    memory_fit = fitness.copy()

    for _ in range(gens):
        for i in range(pop_size):
            if random.random() > AP:
                r = random.randint(0, pop_size-1)
                step = fl * random.random() * (memory[r] - crows[i])
                new_pos = crows[i] + step
            else:
                new_pos = [random.randint(0, w//3), random.randint(0, h//3),
                           random.randint(w//4, w//2), random.randint(h//4, h//2)]

            new_pos = np.clip(new_pos, [0,0,50,50], [w-50, h-50, w//2, h//2]).astype(int)
            reg = Region(*new_pos)
            new_fit = fitness_func(reg, gray)

            if new_fit > fitness[i]:
                crows[i] = new_pos
                fitness[i] = new_fit
                memory[i] = new_pos
                memory_fit[i] = new_fit

    best_idx = np.argmax(memory_fit)
    return Region(*memory[best_idx].astype(int))


# ====================== 4. SCA - Sine Cosine Algorithm ======================
def sinecosine_optimize(leaves, gray, fitness_func, pop_size=30, gens=40):
    h, w = gray.shape
    X = np.array([[random.randint(0, w//3), random.randint(0, h//3),
                   random.randint(w//4, w//2), random.randint(h//4, h//2)] for _ in range(pop_size)])
    fitness = np.array([fitness_func(Region(*x), gray) for x in X])
    best_idx = np.argmax(fitness)
    Xbest = X[best_idx].copy()

    for t in range(gens):
        a = 2
        r1 = a - t * (a / gens)
        for i in range(pop_size):
            for j in range(4):
                r2 = 2 * np.pi * random.random()
                r3 = 2 * random.random()
                r4 = random.random()
                if r4 < 0.5:
                    X[i][j] += r1 * np.sin(r2) * abs(r3 * Xbest[j] - X[i][j])
                else:
                    X[i][j] += r1 * np.cos(r2) * abs(r3 * Xbest[j] - X[i][j])
            X[i] = np.clip(X[i], [0,0,50,50], [w-50, h-50, w//2, h//2]).astype(int)
            reg = Region(*X[i])
            fit = fitness_func(reg, gray)
            if fit > fitness[i]:
                fitness[i] = fit
                if fit > fitness[best_idx]:
                    best_idx = i
                    Xbest = X[i].copy()

    return Region(*Xbest.astype(int))


# ====================== 5. NRO - Newton-Raphson Optimizer (traducción del MATLAB) ======================
def nro_optimize(leaves, gray, fitness_func, N=30, MaxIt=40):
    h, w = gray.shape
    dim = 4
    LB = np.array([0, 0, 50, 50])
    UB = np.array([w//2, h//2, w-10, h-10])

    # Inicialización
    Position = np.random.uniform(LB, UB, (N, dim))
    Fitness = np.array([fitness_func(Region(*np.round(pos).astype(int)), gray) for pos in Position])
    best_idx = np.argmax(Fitness)
    Best_Pos = Position[best_idx].copy()
    Best_Score = Fitness[best_idx]
    Worst_idx = np.argmin(Fitness)
    Worst_Pos = Position[Worst_idx]

    for it in range(MaxIt):
        delta = (1 - (2 * it / MaxIt))**5

        for i in range(N):
            a1, a2 = random.sample(range(N), 2)
            rho = random.random() * (Best_Pos - Position[i]) + random.random() * (Position[a1] - Position[a2])

            # Newton-Raphson Search Rule aproximado
            NRSR = 0.01 * rho  # simplificación numérica
            X1 = Position[i] - NRSR + rho
            X2 = Best_Pos - NRSR + rho
            X3 = Position[i] - delta * (X2 - X1)
            a1, a2 = random.random(), random.random()
            Xupdate = a1 * (a1 * X1 + (1 - a2) * X2) + (1 - a2) * X3

            # Trap Avoidance Operator
            if random.random() < 0.6:
                theta1 = -1 + 2 * random.random()
                theta2 = -0.5 + random.random()
                beta = random.random() < 0.5
                u1 = 3 * random.random() if beta else 1
                u2 = random.random() if beta else 1
                mean_pos = np.mean(Position, axis=0)
                X_TAO = Xupdate + theta1 * (u1 * Best_Pos - u2 * Position[i]) + theta2 * delta * (u1 * mean_pos - u2 * Position[i])
                Xnew = X_TAO
            else:
                Xnew = Xupdate

            Xnew = np.clip(Xnew, LB, UB)
            reg = Region(*np.round(Xnew).astype(int))
            Xnew_Cost = fitness_func(reg, gray)

            if Xnew_Cost > Fitness[i]:
                Position[i] = Xnew
                Fitness[i] = Xnew_Cost
                if Xnew_Cost > Best_Score:
                    Best_Score = Xnew_Cost
                    Best_Pos = Xnew.copy()

    return Region(*np.round(Best_Pos).astype(int))


# ====================== 6. GASDI - Adaptación del MATLAB original ======================
def gasdi_optimize(leaves, gray, fitness_func, N=30, max_iteration=40):
    h, w = gray.shape
    LowB = np.array([0, 0, 50, 50])
    UperB = np.array([w//2, h//2, w-50, h-50])
    
    r = np.random.uniform(LowB, UperB, (N, 4))
    availability = np.array([fitness_func(Region(*np.round(pos).astype(int)), gray) for pos in r])
    pBest = r[np.argmax(availability)].copy()
    pBest_availability = np.max(availability)
    R = np.linalg.norm(UperB - LowB) * 0.3  # radio de vecinos
    e = 0.1
    gama = 0.05

    for _ in range(max_iteration):
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(r - r[i], axis=1)
            neigh_idx = np.where((dists <= R) & (dists > 0))[0]
            neighbors.append(neigh_idx)

        x = np.zeros(N)
        for i in range(N):
            if len(neighbors[i]) > 0:
                x[i] = (1 - e) * availability[i] + (e / len(neighbors[i])) * np.sum(availability[neighbors[i]])
            else:
                x[i] = availability[i]

        for i in range(N):
            if len(neighbors[i]) > 0:
                suma_r = np.zeros(4)
                for j in neighbors[i]:
                    diff = r[j] - r[i]
                    norma = np.linalg.norm(diff)
                    if norma > 1e-6:
                        suma_r += diff / norma
                r[i] = pBest + gama * suma_r * (x[i] * np.sum(x[neighbors[i]]))
            else:
                r[i] += 0.1 * (pBest - r[i]) + 0.01 * np.random.randn(4)
            
            r[i] = np.clip(r[i], LowB, UperB)
            reg = Region(*np.round(r[i]).astype(int))
            fit = fitness_func(reg, gray)
            if fit > availability[i]:
                availability[i] = fit
                if fit > pBest_availability:
                    pBest_availability = fit
                    pBest = r[i].copy()

    return Region(*np.round(pBest).astype(int))


    # ====================== 7. QIO - Quadratic Interpolation Optimization ======================
def qio_optimize(leaves, gray, fitness_func, nPop=30, MaxIt=40):
    h, w = gray.shape
    LB = [0, 0, 50, 50]
    UB = [w//2, h//2, w-50, h-50]
    
    def GQI(x1, x2, x3, f1, f2, f3, L, U):
        denom = (x1-x2)*(f1-f3) - (x1-x3)*(f1-f2) + 1e-8
        xb = 0.5 * ((x1**2-x2**2)*(f1-f3) - (x1**2-x3**2)*(f1-f2)) / denom
        return np.clip(xb, L, U)
    
    X = np.random.uniform(LB, UB, (nPop, 4))
    Fit = np.array([fitness_func(_to_region(x, (h,w)), gray) for x in X])
    best_idx = np.argmax(Fit)
    BestPos = X[best_idx].copy()
    
    for t in range(MaxIt):
        for i in range(nPop):
            idxs = [j for j in range(nPop) if j != i]
            k1, k2, k3 = random.sample(idxs, 3)
            
            if random.random() > 0.5:  # Exploración
                new_pos = np.zeros(4)
                for j in range(4):
                    new_pos[j] = GQI(X[i,j], X[k1,j], X[k2,j], Fit[i], Fit[k1], Fit[k2], LB[j], UB[j])
                w1 = 3 * np.cos(np.pi/2 * t / MaxIt) * np.random.randn(4)
                new_pos += w1 * (X[k3] - new_pos)
            else:  # Explotación
                new_pos = np.zeros(4)
                for j in range(4):
                    new_pos[j] = GQI(X[k1,j], X[k2,j], BestPos[j], Fit[k1], Fit[k2], Fit.max(), LB[j], UB[j])
                w2 = 3 * (1 - t/MaxIt) * np.random.randn(4)
                new_pos += w2 * (BestPos - X[i])
            
            new_pos = np.clip(new_pos, LB, UB)
            new_fit = fitness_func(_to_region(new_pos, (h,w)), gray)
            if new_fit > Fit[i]:
                X[i] = new_pos
                Fit[i] = new_fit
                if new_fit > Fit.max():
                    BestPos = new_pos.copy()
    
    return _to_region(BestPos, (h,w))

# ====================== 8. KLA - Kirchhoff’s Law Algorithm (CORREGIDO) ======================
def kla_optimize(leaves, gray, fitness_func, nPop=35, MaxIt=40):
    h, w = gray.shape
    LB = np.array([0, 0, 50, 50])
    UB = np.array([w//2, h//2, w-50, h-50])
    
    pos = np.random.uniform(LB, UB, (nPop, 4))
    cost = np.array([fitness_func(_to_region(p, (h,w)), gray) for p in pos])
    
    best_idx = np.argmax(cost)
    best_pos = pos[best_idx].copy()  # ← ya existe desde el inicio

    for _ in range(MaxIt):
        for i in range(nPop):
            idxs = [j for j in range(nPop) if j != i]
            a, b, c = random.sample(idxs, 3)
            
            q  = (cost[i] - cost[c]) / (abs(cost[i] - cost[c]) + 1e-12)
            Q1 = (cost[i] - cost[a]) / (abs(cost[i] - cost[a]) + 1e-12)
            Q2 = (cost[i] - cost[b]) / (abs(cost[i] - cost[b]) + 1e-12)
            
            S = (np.random.rand(4) + np.random.rand(4)) * (
                q  * (pos[c] - pos[i]) +
                Q1 * (pos[a] - pos[i]) +
                Q2 * (pos[b] - pos[i])
            )
            
            new_pos = pos[i] + S
            new_pos = np.clip(new_pos, LB, UB)
            new_cost = fitness_func(_to_region(new_pos, (h,w)), gray)
            
            if new_cost > cost[i]:
                pos[i] = new_pos
                cost[i] = new_cost
                if new_cost > cost.max():
                    best_pos = new_pos.copy()   # ← CORREGIDO: best_pos, no best

    return _to_region(best_pos, (h,w))

# ====================== 9. LISSAJOU-DE - Differential Evolution con curva de Lissajou ======================
# Traducción/adaptación del MATLAB (DELissajouCrosSelect.m + InvertedRouletteWheelSelectionCum.m
# + crossover_continious.m). El original minimiza Cost; aquí se maximiza fitness_func.
def _cx_continua(p1, p2, Pc, lb, ub):
    """Cruce continuo (S. Mirjalili) adaptado a vectores numpy."""
    beta = np.random.rand(p1.shape[0])
    c1 = beta*p1 + (1-beta)*p2
    c2 = (1-beta)*p1 + beta*p2
    c1 = np.clip(c1, lb, ub)
    c2 = np.clip(c2, lb, ub)
    if random.random() > Pc:
        c1 = p1.copy()
    if random.random() > Pc:
        c2 = p2.copy()
    return c1, c2


def _dx(pop_selection, Var_min, Var_max, p_best, pc, pcv, pd, alpha):
    """Cruce Direccional (DX) - Das & Pratihar (2018), traducción fiel de dx.m.
    Recibe una poblacion (N x n_var), cruza parejas con probabilidad pc y
    devuelve la poblacion despues del cruce."""
    pop_selection = np.asarray(pop_selection, dtype=float)
    N, n_var = pop_selection.shape
    pop_crossover = np.zeros_like(pop_selection)
    if N % 2 != 0:
        N = N - 1
    p = np.random.permutation(N)
    for idx in range(0, N, 2):
        p1, p2 = p[idx], p[idx + 1]
        parent1 = pop_selection[p1]
        parent2 = pop_selection[p2]
        child1 = np.zeros(n_var)
        child2 = np.zeros(n_var)
        if random.random() <= pc:  # la pareja se cruza con probabilidad pc
            for i in range(n_var):
                if random.random() <= pcv:
                    y3 = p_best[i]
                    if parent1[i] != parent2[i]:
                        y1 = parent1[i]
                        y2 = parent2[i]
                        yl = Var_min
                        yu = Var_max
                        rand_var = random.random()
                        val = 1 - (0.5 ** (np.exp(abs(y2 - y1) / (yu - yl))))
                        pmean = 0.5 * (y1 + y2)
                        beta = rand_var / (alpha ** 2)
                        if y3 >= pmean:
                            if random.random() <= pd:
                                c1 = val*(y1+y2) + (alpha**rand_var)*(np.exp(1-beta))*(1-val)*abs(y2-y1)
                                c2 = (1-val)*(y1+y2) - (alpha**(1-rand_var))*np.exp(-beta)*val*abs(y2-y1)
                            else:
                                c1 = val*(y1+y2) - (alpha**rand_var)*(np.exp(1-beta))*(1-val)*abs(y2-y1)
                                c2 = (1-val)*(y1+y2) + (alpha**(1-rand_var))*np.exp(-beta)*val*abs(y2-y1)
                        else:
                            if random.random() <= pd:
                                c1 = val*(y1+y2) - (alpha**rand_var)*(np.exp(1-beta))*(1-val)*abs(y2-y1)
                                c2 = (1-val)*(y1+y2) + (alpha**(1-rand_var))*np.exp(-beta)*val*abs(y2-y1)
                            else:
                                c1 = val*(y1+y2) + (alpha**rand_var)*(np.exp(1-beta))*(1-val)*abs(y2-y1)
                                c2 = (1-val)*(y1+y2) - (alpha**(1-rand_var))*np.exp(-beta)*val*abs(y2-y1)
                        c1 = min(max(c1, yl), yu)
                        c2 = min(max(c2, yl), yu)
                        if random.random() <= 0.5:
                            child1[i] = c2
                            child2[i] = c1
                        else:
                            child1[i] = c1
                            child2[i] = c2
                    else:
                        pmean = parent1[i]
                        if y3 != pmean:
                            yl = Var_min
                            yu = Var_max
                            rand_var = random.random()
                            beta = rand_var / (alpha ** 2)
                            val = 1 - (0.5 ** (np.exp(abs(pmean - y3) / (yu - yl))))
                            if random.random() <= pd:
                                c1 = val*(y3+pmean) + (alpha**rand_var)*(np.exp(1-beta))*(1-val)*(y3-pmean)
                                c2 = (1-val)*(y3+pmean) - (alpha**(1-rand_var))*np.exp(-beta)*val*(y3-pmean)
                            else:
                                c1 = val*(y3+pmean) - (alpha**rand_var)*(np.exp(1-beta))*(1-val)*(y3-pmean)
                                c2 = (1-val)*(y3+pmean) + (alpha**(1-rand_var))*np.exp(-beta)*val*(y3-pmean)
                            c1 = min(max(c1, yl), yu)
                            c2 = min(max(c2, yl), yu)
                            if random.random() <= 0.5:
                                child1[i] = c2
                                child2[i] = c1
                            else:
                                child1[i] = c1
                                child2[i] = c2
                        else:
                            child1[i] = parent1[i]
                            child2[i] = parent2[i]
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        pop_crossover[idx] = child1
        pop_crossover[idx + 1] = child2
    return pop_crossover


def _ruleta_fitness(fits):
    """Ruleta proporcional al fitness (equivalente a la ruleta invertida del MATLAB,
    que favorecía al menor Cost; aquí favorece al mayor fitness)."""
    w = np.array(fits, dtype=float)
    w = w - w.min() + 1e-12
    if w.sum() <= 0:
        return random.randrange(len(fits))
    return int(np.searchsorted(np.cumsum(w/w.sum()), random.random()))


def lissajou_optimize(leaves, gray, fitness_func, pop_size=30, gens=40):
    h, w = gray.shape
    D = 4  # x, y, w, h
    LB = np.array([0, 0, 50, 50], dtype=float)
    UB = np.array([w//2, h//2, w-50, h-50], dtype=float)

    # Parámetros del MATLAB
    F = 0.8          # factor de escalado (beta)
    pCR = 0.2        # probabilidad de cruce binomial DE
    Delta = np.pi/2  # fase de la curva de Lissajou
    alpha = 0.95     # factor multiplicativo del cruce direccional
    pcv = 0.95       # probabilidad variable-a-variable del cruce direccional
    pd = 0.5         # probabilidad direccional
    Pc = 0.2         # probabilidad del cruce continuo

    def fit(pos):
        return fitness_func(_to_region(pos, (h, w)), gray)

    # Inicialización
    X = np.random.uniform(LB, UB, (pop_size, D))
    Fit = np.array([fit(x) for x in X])
    best_idx = np.argmax(Fit)
    BestPos = X[best_idx].copy()
    BestFit = Fit[best_idx]

    for it in range(gens):
        # a2 disminuye linealmente de -1 a -2 (parámetro de la espiral)
        a2 = -1 + it*((-1)/gens)

        for i in range(pop_size):
            x = X[i].copy()
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = random.sample(idxs, 3)
            l = (a2 - 1)*random.random() + 1

            # Mutación rand/1 con curva de Lissajou
            espiral = np.zeros(D)
            for j in range(D):
                dist = abs(X[b, j] - X[c, j])
                if random.random() > 0.5:
                    espiral[j] = dist*np.sin(6*l*np.pi + Delta)
                else:
                    espiral[j] = dist*np.sin(4*l*np.pi)
            y = X[a] + F*espiral
            y = np.clip(y, LB, UB)

            # 1) Cruce binomial clásico DE
            z = x.copy()
            j0 = random.randrange(D)
            for j in range(D):
                if j == j0 or random.random() <= pCR:
                    z[j] = y[j]
            fit_cx = fit(z)

            # 2) Cruce continuo
            c1, c2 = _cx_continua(y, x, Pc, LB, UB)
            f1, f2 = fit(c1), fit(c2)
            ccx_pos, ccx_fit = (c1, f1) if f1 >= f2 else (c2, f2)

            # 3) Cruce direccional (DX de Das & Pratihar)
            NewCrossParticle = _dx(np.vstack([y, x]), LB[0], UB[0], BestPos, pCR, pcv, pd, alpha)
            f1, f2 = fit(NewCrossParticle[0]), fit(NewCrossParticle[1])
            dx_pos, dx_fit = (NewCrossParticle[0], f1) if f1 >= f2 else (NewCrossParticle[1], f2)

            # Selección por ruleta (invertida) entre los 3 descendientes
            cands = [z, ccx_pos, dx_pos]
            fits = [fit_cx, ccx_fit, dx_fit]
            sel = _ruleta_fitness(fits)
            NewPos = cands[sel]
            NewFit = fits[sel]

            if NewFit > BestFit:
                BestFit = NewFit
                BestPos = NewPos.copy()

            # Selección adaptativa: fitness + distancia al mejor
            fmax, fmin = Fit.max(), Fit.min()
            dists = np.linalg.norm(X - BestPos, axis=1)
            Dis_max = dists.max()
            Dis_cur = dists[i]
            Dis_xu = np.linalg.norm(NewPos - BestPos)
            alfa = 0.8 + (1 - 0.8)*random.random()

            a1 = alfa*((fmax - Fit[i])/(fmax - fmin + 1e-12))
            a2 = (1 - alfa)*((Dis_max - Dis_cur)/(Dis_max + Dis_cur + 1e-12))
            Fwi = a1 + a2
            b1 = alfa*((fmax - NewFit)/(fmax - fmin + 1e-12))
            b2 = (1 + alfa)*((Dis_max - Dis_xu)/(Dis_max + Dis_cur + 1e-12))
            Fwu = b1 + b2

            if NewFit > Fit[i] or (Fwu <= Fwi and Fit[i] != BestFit):
                X[i] = NewPos
                Fit[i] = NewFit

    return _to_region(BestPos, (h, w))
