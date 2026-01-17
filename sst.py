import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Configurazione
N_REP = 4  # Numero di repliche per lato

console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Funzioni numeriche
def numeric_sinc(x):
    """Gestisce il limite notevole sin(x)/x -> 1 per x=0."""
    return np.where(np.isclose(x, 0.0), 1.0, np.sin(x) / x)

def numeric_fallback(*args):
    """Evita il crash del grafico se un integrale simbolico fallisce."""
    try: return np.zeros_like(args[0], dtype=float)
    except: return 0.0

class PeriodicSignal:
    def __init__(self, base_signal, period, terms):
        self.base_signal = base_signal
        self.period = period
        self.terms = terms

# Legenda
def print_user_guide():
    console.print()
    console.rule("[bold cyan]ANALIZZATORE SPETTRALE[/bold cyan]")
    
    table = Table(title="Guida Comandi Completa", border_style="blue", header_style="bold magenta")
    
    # Colonne
    table.add_column("Categoria", style="yellow")
    table.add_column("Comando", style="green")
    table.add_column("Esempio / Note", style="white")
    
    # Righe Segnali Base
    table.add_row("Rettangolo", "rect(t/T)", "rect(t/2) -> Larghezza 2")
    table.add_row("Triangolo", "tri(t/T)", "tri(t) -> Semidurata 1")
    table.add_row("Gradino", "u(t)", "u(t) - u(t-1)")
    
    # Righe Operazioni
    table.add_section()
    table.add_row("Ripetizione", "rep(x, T)", "rep(rect(t), 4)")
    table.add_row("Campionamento", "comb(T)", "rect(t) * comb(0.5)")
    table.add_row("Convoluzione", "conv(x, y)", "conv(rect(t), rect(t))")
    table.add_row("Filtraggio", "filter(x, H)", "filter(rect(t), rect(f))")

    # Righe Input
    table.add_section()
    table.add_row("Input iniziale", "x(t) oppure X(f)", "Riconoscimento automatico")

    # Righe Avvio
    table.add_section()
    table.add_row("Avvio", "python main_split.py", "Menu separati tempo/frequenza")
    table.add_row("Avvio", "python main_dynamic.py", "Menu dinamico adattivo")
    
    console.print(table)
    console.print("[yellow]Nota:[/yellow] Usa sempre l'asterisco per moltiplicare (es. [bold]2*T[/bold]).")
    console.print("[yellow]Nota:[/yellow] sst.py e' una libreria; avvia uno dei main.\n")

# Symbolic environment.
def get_symbolic_environment():
    t, f = sp.symbols('t f')
    T, A, f0, T0 = sp.symbols('T A f0 T0', real=True, positive=True)

    def piecewise_rect(x): 
        return sp.Piecewise((1, (x >= -0.5) & (x <= 0.5)), (0, True))
    
    def piecewise_tri(x): 
        return sp.Piecewise(
            (1 + x, (x >= -1) & (x < 0)),
            (1 - x, (x >= 0) & (x <= 1)),
            (0, True)
        )

    def gen_rep(signal, period):
        terms = []
        for k in range(-N_REP, N_REP + 1):
            terms.append(signal.subs(t, t - k * period))
        return PeriodicSignal(signal, period, terms)
    
    def gen_comb(period):
        somma = 0
        for k in range(-N_REP, N_REP + 1):
            somma += sp.DiracDelta(t - k * period)
        return somma

    def gen_conv(f1, f2):
        tau_var = sp.symbols('tau', real=True)
        return sp.integrate(f1.subs(t, tau_var) * f2.subs(t, t - tau_var), (tau_var, -sp.oo, sp.oo))

    # Filtraggio
    def gen_filter(signal_t, transfer_f):
        # 1. Passa in frequenza
        X_f = sp.fourier_transform(signal_t, t, f, noconds=True)
        # 2. Applica il filtro
        Y_f = X_f * transfer_f
        # 3. Antitrasforma per ottenere il segnale filtrato nel tempo
        return sp.inverse_fourier_transform(Y_f, f, t, noconds=True)

    symbol_map = {
        't': t, 'f': f,
        'rect': piecewise_rect, 'tri': piecewise_tri, 'u': sp.Heaviside,
        'sinc': sp.sinc, 'pi': sp.pi, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 
        'abs': sp.Abs, 'j': sp.I, 'DiracDelta': sp.DiracDelta,
        'rep': gen_rep, 'comb': gen_comb, 'conv': gen_conv,
        'filter': gen_filter,
        'T': T, 'A': A, 'f0': f0, 'T0': T0
    }
    default_values = {T: 1, T0: 1, A: 1, f0: 1}
    return t, f, symbol_map, default_values

# Expression formatting.
def _extract_var_bounds(cond, symbol_hint):
    if not isinstance(cond, sp.And):
        return None
    var_expr = None
    bounds = []
    for rel in cond.args:
        if not isinstance(rel, sp.Relational):
            return None
        lhs, rhs = rel.lhs, rel.rhs
        if lhs.has(symbol_hint) and rhs.is_number:
            expr_var = lhs
            bound = rhs
            sense = rel.rel_op
        elif rhs.has(symbol_hint) and lhs.is_number:
            expr_var = rhs
            bound = lhs
            sense = rel.rel_op
            if sense == "<=":
                sense = ">="
            elif sense == "<":
                sense = ">"
            elif sense == ">=":
                sense = "<="
            elif sense == ">":
                sense = "<"
        else:
            return None
        if var_expr is None:
            var_expr = expr_var
        else:
            if sp.simplify(var_expr - expr_var) != 0:
                return None
        bounds.append((sense, bound))
    if var_expr is None or len(bounds) != 2:
        return None
    lower = upper = None
    lower_inc = upper_inc = False
    for sense, bound in bounds:
        if sense in (">=", ">"):
            lower = bound
            lower_inc = (sense == ">=")
        elif sense in ("<=", "<"):
            upper = bound
            upper_inc = (sense == "<=")
        else:
            return None
    if lower is None or upper is None:
        return None
    return var_expr, sp.simplify(lower), sp.simplify(upper), lower_inc, upper_inc

def _extract_step_var(cond, symbol_hint):
    if not isinstance(cond, sp.Relational):
        return None
    lhs, rhs = cond.lhs, cond.rhs
    if rhs.is_number and lhs.has(symbol_hint):
        var_expr = lhs
        bound = rhs
        sense = cond.rel_op
    elif lhs.is_number and rhs.has(symbol_hint):
        var_expr = rhs
        bound = lhs
        sense = cond.rel_op
        if sense == "<=":
            sense = ">="
        elif sense == "<":
            sense = ">"
        elif sense == ">=":
            sense = "<="
        elif sense == ">":
            sense = "<"
    else:
        return None
    return var_expr, sp.simplify(bound), sense

def _simplify_piecewise_named(expr, symbol_hint):
    rect_sym = sp.Function('rect')
    tri_sym = sp.Function('tri')
    u_sym = sp.Function('u')

    def convert_piecewise(pw):
        if not isinstance(pw, sp.Piecewise):
            return pw
        if len(pw.args) < 2:
            return pw

        # Step: 1 on [0, +inf), 0 otherwise.
        if sp.simplify(first_expr - 1) == 0 and last_expr == 0 and last_cond is True:
            step = _extract_step_var(first_cond, symbol_hint)
            if step:
                var_expr, bound, sense = step
                if sp.simplify(bound) == 0 and sense in (">=", ">"):
                    return u_sym(var_expr)

        # Rectangular: 1 on [c - T/2, c + T/2], 0 otherwise.
        first_expr, first_cond = pw.args[0]
        last_expr, last_cond = pw.args[-1]
        if sp.simplify(first_expr - 1) == 0 and last_expr == 0 and last_cond is True:
            bounds = _extract_var_bounds(first_cond, symbol_hint)
            if bounds:
                var_expr, lower, upper, _linc, _uinc = bounds
                mid = sp.simplify((lower + upper) / 2)
                width = sp.simplify(upper - lower)
                if sp.simplify(width) != 0:
                    scaled = sp.simplify((var_expr - mid) / width)
                    return rect_sym(scaled)

        # Triangular: 1+(v-c)/T on [c-T,c], 1-(v-c)/T on [c,c+T], 0 otherwise.
        if len(pw.args) >= 3 and last_expr == 0 and last_cond is True:
            (expr1, cond1), (expr2, cond2) = pw.args[0], pw.args[1]
            b1 = _extract_var_bounds(cond1, symbol_hint)
            b2 = _extract_var_bounds(cond2, symbol_hint)
            if b1 and b2:
                var1, l1, u1, _linc1, _uinc1 = b1
                var2, l2, u2, _linc2, _uinc2 = b2
                if sp.simplify(var1 - var2) == 0:
                    v = var1
                    mid = sp.simplify((u1 + l2) / 2)
                    T = sp.simplify(u2 - l2)
                    if sp.simplify(u1 - l2) == 0 and sp.simplify(l1 - (mid - T)) == 0 and sp.simplify(u2 - (mid + T)) == 0:
                        expr1_expected = sp.simplify(1 + (v - mid) / T)
                        expr2_expected = sp.simplify(1 - (v - mid) / T)
                        if sp.simplify(expr1 - expr1_expected) == 0 and sp.simplify(expr2 - expr2_expected) == 0:
                            return tri_sym(sp.simplify((v - mid) / T))

        return pw

    return expr.replace(sp.Piecewise, convert_piecewise)

def make_readable(expr):
    expr = _simplify_piecewise_named(expr, sp.Symbol('t'))
    expr = _simplify_piecewise_named(expr, sp.Symbol('f'))
    visual_exp = sp.Function('exp')
    expr_clean = expr.replace(sp.exp, visual_exp)
    visual_sinc = sp.Function('sinc')
    expr_clean = expr_clean.replace(sp.sinc, lambda arg: visual_sinc(sp.simplify(arg / sp.pi)))
    return expr_clean

def smart_simplify(expr, f_symbol):
    # 1. Espansione
    expr = sp.expand(expr)
    
    # 2. Rimozione Fase
    expr_exp = expr.rewrite(sp.exp).expand()
    exponentials = [a.args[0] for a in expr_exp.atoms(sp.exp) if a.args[0].has(f_symbol)]
    center_term = 1
    
    if exponentials:
        coeffs = [arg.coeff(f_symbol) for arg in exponentials if arg.coeff(f_symbol)]
        if coeffs:
            avg_coeff = sum(coeffs) / len(coeffs)
            center_term = sp.exp(avg_coeff * f_symbol)
            expr = sp.simplify(expr / center_term)

    # 3. Semplificazione Trigonometrica (Cruciale per sinc^2)
    expr = sp.trigsimp(expr)
    
    # 4. Conversione in Sinc
    expr = expr.rewrite(sp.sin)
    expr = sp.simplify(expr)
    expr = expr.rewrite(sp.sinc)
    
    return make_readable(center_term * expr)

# Zoom automatico
def calculate_smart_limits(expr, t_symbol, defaults):
    try:
        safe_modules = [{'sinc': numeric_sinc, 'FourierTransform': numeric_fallback}, 'numpy']
        func = sp.lambdify(t_symbol, expr.subs(defaults), modules=safe_modules)
        test_t = np.linspace(-50, 50, 5000)
        y = func(test_t)
        y_abs = np.abs(y)
        nonzero = np.where(y_abs > 0.01 * np.max(y_abs + 1e-9))[0]
        
        if len(nonzero) > 0:
            duration = test_t[nonzero[-1]] - test_t[nonzero[0]]
            duration = max(min(duration, 20), 0.1)
            t_lim = max(duration * 1.5, 3.0)
            f_lim = 4.0 * (1.0 / (duration / 2.0)) 
            return t_lim, f_lim
    except:
        pass
    return 6.0, 2.0 

# Frequency zoom refinement.
def refine_frequency_limit(freq_func, f_lim):
    probe_lim = max(f_lim * 4, 10.0)
    f_probe = np.linspace(0, probe_lim, 16000)
    f_probe[0] = 1e-6
    try:
        y_probe = freq_func(f_probe)
        if np.isscalar(y_probe): y_probe = np.full_like(f_probe, y_probe)
        y_probe = np.real_if_close(y_probe, tol=1000)
        if np.iscomplexobj(y_probe): y_probe = np.real(y_probe)
    except:
        return f_lim

    y_abs = np.abs(y_probe)
    max_y = np.max(y_abs + 1e-12)
    eps = 0.01 * max_y
    minima = (y_abs[1:-1] <= y_abs[:-2]) & (y_abs[1:-1] <= y_abs[2:]) & (y_abs[1:-1] < eps)
    idx = np.where(minima)[0] + 1

    zeros = []
    min_sep = (f_probe[1] - f_probe[0]) * 8
    for i in idx:
        f0 = f_probe[i]
        if f0 == 0.0:
            continue
        if not zeros or (f0 - zeros[-1]) > min_sep:
            zeros.append(f0)
        if len(zeros) >= 3:
            break

    if len(zeros) >= 3:
        return min(f_lim, zeros[2] * 1.05)
    return f_lim

# Combined time/frequency plotting.
def plot_signals(t, f, time_func, freq_func, t_lim, f_lim):
    f_lim = refine_frequency_limit(freq_func, f_lim)
    console.print(f"[cyan]Zoom Dinamico: Tempo +/- {t_lim:.1f}s | Frequenza +/- {f_lim:.1f} Hz[/cyan]")
    
    t_vals = np.linspace(-t_lim, t_lim, 3000) 
    f_vals = np.linspace(-f_lim, f_lim, 3000) + 1e-6 

    try:
        y_time = time_func(t_vals)
        if np.isscalar(y_time): y_time = np.full_like(t_vals, y_time)
        y_time = np.real_if_close(y_time, tol=1000)
        if np.iscomplexobj(y_time): y_time = np.real(y_time)
    except: y_time = np.zeros_like(t_vals)

    try:
        y_freq = freq_func(f_vals)
        if np.isscalar(y_freq): y_freq = np.full_like(f_vals, y_freq)
        y_freq = np.real_if_close(y_freq, tol=1000)
        if np.iscomplexobj(y_freq): y_freq = np.real(y_freq)
    except: y_freq = np.zeros_like(f_vals)

    plt.figure(figsize=(12, 10))

    # Plot Tempo
    plt.subplot(2, 1, 1)
    plt.plot(t_vals, y_time, 'b-', linewidth=2)
    plt.title('Dominio del Tempo x(t)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black')
    plt.xlim(-t_lim, t_lim)

    # Plot Frequenza
    plt.subplot(2, 1, 2)
    plt.plot(f_vals, y_freq, 'r-', linewidth=2)
    plt.title('Spettro Reale X(f)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black')
    plt.xlim(-f_lim, f_lim)
    
    step = f_lim / 8
    if step < 0.15: step = 0.1
    elif step < 0.35: step = 0.25
    elif step < 0.75: step = 0.5
    else: step = 1.0
    plt.xticks(np.arange(-f_lim, f_lim + step/10, step)) 

    plt.tight_layout()
    plt.show()

# Plot helpers.
# Plot helper for overlaying two frequency responses.
def plot_frequency_pair(f, freq_func_a, freq_func_b, f_lim, label_a, label_b):
    f_vals = np.linspace(-f_lim, f_lim, 3000) + 1e-6
    try:
        y_a = freq_func_a(f_vals)
        if np.isscalar(y_a): y_a = np.full_like(f_vals, y_a)
        y_a = np.real_if_close(y_a, tol=1000)
        if np.iscomplexobj(y_a): y_a = np.real(y_a)
    except:
        y_a = np.zeros_like(f_vals)

    try:
        y_b = freq_func_b(f_vals)
        if np.isscalar(y_b): y_b = np.full_like(f_vals, y_b)
        y_b = np.real_if_close(y_b, tol=1000)
        if np.iscomplexobj(y_b): y_b = np.real(y_b)
    except:
        y_b = np.zeros_like(f_vals)

    plt.figure(figsize=(12, 6))
    plt.plot(f_vals, y_a, 'r-', linewidth=2, label=label_a)
    plt.plot(f_vals, y_b, 'b-', linewidth=2, label=label_b)
    plt.title('Spettri in Frequenza')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black')
    plt.xlim(-f_lim, f_lim)
    plt.legend()

    step = f_lim / 8
    if step < 0.15: step = 0.1
    elif step < 0.35: step = 0.25
    elif step < 0.75: step = 0.5
    else: step = 1.0
    plt.xticks(np.arange(-f_lim, f_lim + step/10, step))

    plt.tight_layout()
    plt.show()

# Plot helper for overlaying three frequency responses.
def plot_frequency_triplet(f, freq_func_a, freq_func_b, freq_func_c, f_lim, label_a, label_b, label_c):
    f_vals = np.linspace(-f_lim, f_lim, 3000) + 1e-6
    def eval_freq(func):
        try:
            y = func(f_vals)
            if np.isscalar(y): y = np.full_like(f_vals, y)
            y = np.real_if_close(y, tol=1000)
            if np.iscomplexobj(y): y = np.real(y)
            return y
        except:
            return np.zeros_like(f_vals)

    y_a = eval_freq(freq_func_a)
    y_b = eval_freq(freq_func_b)
    y_c = eval_freq(freq_func_c)

    plt.figure(figsize=(12, 6))
    plt.plot(f_vals, y_a, 'r-', linewidth=2, label=label_a)
    plt.plot(f_vals, y_b, 'b-', linewidth=2, label=label_b)
    plt.plot(f_vals, y_c, 'g-', linewidth=2, label=label_c)
    plt.title('Spettri in Frequenza')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black')
    plt.xlim(-f_lim, f_lim)
    plt.legend()

    step = f_lim / 8
    if step < 0.15: step = 0.1
    elif step < 0.35: step = 0.25
    elif step < 0.75: step = 0.5
    else: step = 1.0
    plt.xticks(np.arange(-f_lim, f_lim + step/10, step))

    plt.tight_layout()
    plt.show()

# Plot helper for overlaying two time-domain signals.
def plot_time_pair(t, time_func_a, time_func_b, t_lim, label_a, label_b):
    t_vals = np.linspace(-t_lim, t_lim, 3000)
    def eval_time(func):
        try:
            y = func(t_vals)
            if np.isscalar(y): y = np.full_like(t_vals, y)
            y = np.real_if_close(y, tol=1000)
            if np.iscomplexobj(y): y = np.real(y)
            return y
        except:
            return np.zeros_like(t_vals)

    y_a = eval_time(time_func_a)
    y_b = eval_time(time_func_b)

    plt.figure(figsize=(12, 6))
    plt.plot(t_vals, y_a, 'r-', linewidth=2, label=label_a)
    plt.plot(t_vals, y_b, 'b-', linewidth=2, label=label_b)
    plt.title('Segnali nel Tempo')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black')
    plt.xlim(-t_lim, t_lim)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot helper for overlaying three time-domain signals.
def plot_time_triplet(t, time_func_a, time_func_b, time_func_c, t_lim, label_a, label_b, label_c):
    t_vals = np.linspace(-t_lim, t_lim, 3000)
    def eval_time(func):
        try:
            y = func(t_vals)
            if np.isscalar(y): y = np.full_like(t_vals, y)
            y = np.real_if_close(y, tol=1000)
            if np.iscomplexobj(y): y = np.real(y)
            return y
        except:
            return np.zeros_like(t_vals)

    y_a = eval_time(time_func_a)
    y_b = eval_time(time_func_b)
    y_c = eval_time(time_func_c)

    plt.figure(figsize=(12, 6))
    plt.plot(t_vals, y_a, 'r-', linewidth=2, label=label_a)
    plt.plot(t_vals, y_b, 'b-', linewidth=2, label=label_b)
    plt.plot(t_vals, y_c, 'g-', linewidth=2, label=label_c)
    plt.title('Segnali nel Tempo')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black')
    plt.xlim(-t_lim, t_lim)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Input & domain helpers.
def eval_user_expr(prompt, symbol_map):
    raw_input = console.input(prompt)
    return eval(raw_input.replace('^', '**'), symbol_map)

def get_terms_and_base(obj):
    if isinstance(obj, PeriodicSignal):
        return obj.terms, obj.base_signal
    if isinstance(obj, list):
        return obj, sum(obj)
    return [obj], obj

def normalize_expr(expr):
    return sum(expr) if isinstance(expr, list) else expr

# Infer domain from symbols: t -> time, f -> frequency, else ambiguous.
def infer_domain(expr, t_symbol, f_symbol):
    try:
        free_syms = expr.free_symbols
    except Exception:
        return "amb"
    has_t = t_symbol in free_syms
    has_f = f_symbol in free_syms
    if has_t and not has_f:
        return "t"
    if has_f and not has_t:
        return "f"
    return "amb"

def resolve_domain(expr, t_symbol, f_symbol):
    domain = infer_domain(expr, t_symbol, f_symbol)
    if domain == "amb":
        choice = console.input("[bold green]Ambiguo: 1) x(t) 2) X(f): [/bold green]").strip()
        domain = "t" if choice == "1" else "f"
    return domain

def prompt_initial_expr_auto(t, f, symbol_map):
    expr = eval_user_expr("[bold green]Inserisci x(t) o X(f): [/bold green]", symbol_map)
    expr = normalize_expr(expr)
    return expr

def resolve_filter(t, f, symbol_map):
    raw_filter = eval_user_expr("[bold green]Inserisci h(t) o H(f): [/bold green]", symbol_map)
    filter_expr = normalize_expr(raw_filter)
    domain = infer_domain(filter_expr, t, f)
    if domain == "t":
        h_t = filter_expr
        H_f = sp.fourier_transform(h_t, t, f, noconds=True)
        return h_t, H_f, False
    if domain == "f":
        H_f = filter_expr
        h_t = sp.inverse_fourier_transform(H_f, f, t, noconds=True)
        return h_t, H_f, True
    choice = console.input("[bold green]Ambiguo: 1) H(f) 2) h(t): [/bold green]").strip()
    if choice == "2":
        h_t = filter_expr
        H_f = sp.fourier_transform(h_t, t, f, noconds=True)
        return h_t, H_f, False
    H_f = filter_expr
    h_t = sp.inverse_fourier_transform(H_f, f, t, noconds=True)
    return h_t, H_f, True

# Transform helpers.
def compute_total_ft(terms, t, f):
    X_tot = 0
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.percentage:>3.0f}%")) as p:
        task = p.add_task("Calcolo Trasformata...", total=len(terms))
        for term in terms:
            X_tot += sp.fourier_transform(term, t, f, noconds=True)
            p.advance(task)
    return X_tot

def get_safe_modules():
    return [{'sinc': numeric_sinc, 'FourierTransform': numeric_fallback}, 'numpy']

def transform_auto(expr, domain, t, f, defaults):
    safe_modules = get_safe_modules()
    if domain == "t":
        terms, base = get_terms_and_base(expr)
        t_lim, f_lim = calculate_smart_limits(base, t, defaults)
        x_total = sum(terms)
        res_str = sp.pretty(smart_simplify(x_total, t), use_unicode=True)
        console.print(Panel(res_str, title="Risultato x(t)", border_style="green"))

        X_tot = compute_total_ft(terms, t, f)
        res_str = sp.pretty(smart_simplify(X_tot, f), use_unicode=True)
        console.print(Panel(res_str, title="Risultato X(f)", border_style="green"))

        time_f = sp.lambdify(t, x_total.subs(defaults), modules=safe_modules)
        freq_f = sp.lambdify(f, sp.re(X_tot).subs(defaults), modules=safe_modules)
        plot_signals(t, f, time_f, freq_f, t_lim, f_lim)
        return

    if domain == "f":
        X_expr = normalize_expr(expr)
        res_str = sp.pretty(smart_simplify(X_expr, f), use_unicode=True)
        console.print(Panel(res_str, title="Risultato X(f)", border_style="green"))

        x_expr = sp.inverse_fourier_transform(X_expr, f, t, noconds=True)
        res_str = sp.pretty(smart_simplify(x_expr, t), use_unicode=True)
        console.print(Panel(res_str, title="Risultato x(t)", border_style="green"))

        t_lim, f_lim = calculate_smart_limits(x_expr, t, defaults)
        time_f = sp.lambdify(t, x_expr.subs(defaults), modules=safe_modules)
        freq_f = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
        plot_signals(t, f, time_f, freq_f, t_lim, f_lim)
        return

    console.print("[bold red]Dominio non riconosciuto.[/bold red]")

def transform_auto_infer(expr, t, f, defaults):
    domain = resolve_domain(expr, t, f)
    transform_auto(expr, domain, t, f, defaults)

def ask_yes_no(prompt):
    ans = console.input(prompt).strip().lower()
    return ans != 'n'

# Flow helpers.
def flow_transform_auto(expr, t, f, defaults):
    transform_auto_infer(expr, t, f, defaults)

# Flow helpers.
def flow_filter_frequency_from_time(x_expr, t, f, defaults, symbol_map):
    safe_modules = get_safe_modules()
    terms, base = get_terms_and_base(x_expr)
    X_expr = compute_total_ft(terms, t, f)
    f_lim = calculate_smart_limits(base, t, defaults)[1]

    h_t, H_f, _from_H = resolve_filter(t, f, symbol_map)

    res_str = sp.pretty(smart_simplify(X_expr, f), use_unicode=True)
    console.print(Panel(res_str, title="Risultato X(f)", border_style="green"))
    res_str = sp.pretty(smart_simplify(H_f, f), use_unicode=True)
    console.print(Panel(res_str, title="Risultato H(f)", border_style="green"))
    if ask_yes_no("[bold green]Mostrare grafico di X(f) e H(f) insieme? (s/n): [/bold green]"):
        freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
        freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
        plot_frequency_pair(f, freq_X, freq_H, f_lim, "X(f)", "H(f)")

    Y_f = sp.simplify(X_expr * H_f)
    res_str = sp.pretty(smart_simplify(Y_f, f), use_unicode=True)
    console.print(Panel(res_str, title="Risultato Y(f)", border_style="green"))
    if ask_yes_no("[bold green]Mostrare grafico di X(f), H(f), Y(f) insieme? (s/n): [/bold green]"):
        freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
        freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
        freq_Y = sp.lambdify(f, sp.re(Y_f).subs(defaults), modules=safe_modules)
        plot_frequency_triplet(f, freq_X, freq_H, freq_Y, f_lim, "X(f)", "H(f)", "Y(f)")

def flow_filter_frequency_from_freq(X_expr, t, f, defaults, symbol_map):
    safe_modules = get_safe_modules()
    h_t, H_f, _from_H = resolve_filter(t, f, symbol_map)
    f_lim = calculate_smart_limits(sp.exp(-t**2), t, defaults)[1]

    res_str = sp.pretty(smart_simplify(X_expr, f), use_unicode=True)
    console.print(Panel(res_str, title="Risultato X(f)", border_style="green"))
    res_str = sp.pretty(smart_simplify(H_f, f), use_unicode=True)
    console.print(Panel(res_str, title="Risultato H(f)", border_style="green"))
    if ask_yes_no("[bold green]Mostrare grafico di X(f) e H(f) insieme? (s/n): [/bold green]"):
        freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
        freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
        plot_frequency_pair(f, freq_X, freq_H, f_lim, "X(f)", "H(f)")

    Y_f = sp.simplify(X_expr * H_f)
    res_str = sp.pretty(smart_simplify(Y_f, f), use_unicode=True)
    console.print(Panel(res_str, title="Risultato Y(f)", border_style="green"))
    if ask_yes_no("[bold green]Mostrare grafico di X(f), H(f), Y(f) insieme? (s/n): [/bold green]"):
        freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
        freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
        freq_Y = sp.lambdify(f, sp.re(Y_f).subs(defaults), modules=safe_modules)
        plot_frequency_triplet(f, freq_X, freq_H, freq_Y, f_lim, "X(f)", "H(f)", "Y(f)")

def flow_filter_time_from_time(x_expr, t, f, defaults, symbol_map):
    safe_modules = get_safe_modules()
    h_t, H_f, from_H = resolve_filter(t, f, symbol_map)
    terms, _base = get_terms_and_base(x_expr)
    x_total = sum(terms)

    if from_H:
        res_str = sp.pretty(smart_simplify(x_total, t), use_unicode=True)
        console.print(Panel(res_str, title="Risultato x(t)", border_style="green"))
        res_str = sp.pretty(smart_simplify(h_t, t), use_unicode=True)
        console.print(Panel(res_str, title="Risultato h(t)", border_style="cyan"))

    if ask_yes_no("[bold green]Mostrare grafico doppio di x(t) e h(t)? (s/n): [/bold green]"):
        t_lim_x, _ = calculate_smart_limits(x_total, t, defaults)
        t_lim_h, _ = calculate_smart_limits(h_t, t, defaults)
        t_lim = max(t_lim_x, t_lim_h)
        time_x = sp.lambdify(t, x_total.subs(defaults), modules=safe_modules)
        time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
        plot_time_pair(t, time_x, time_h, t_lim, "x(t)", "h(t)")

    X_expr = compute_total_ft(terms, t, f)
    Y_f = sp.simplify(X_expr * H_f)
    y_t = sp.inverse_fourier_transform(Y_f, f, t, noconds=True)
    res_str = sp.pretty(smart_simplify(y_t, t), use_unicode=True)
    console.print(Panel(res_str, title="Risultato y(t)", border_style="green"))
    if ask_yes_no("[bold green]Mostrare grafico triplo di x(t), h(t), y(t)? (s/n): [/bold green]"):
        t_lim_x, _ = calculate_smart_limits(x_total, t, defaults)
        t_lim_h, _ = calculate_smart_limits(h_t, t, defaults)
        t_lim_y, _ = calculate_smart_limits(y_t, t, defaults)
        t_lim = max(t_lim_x, t_lim_h, t_lim_y)
        time_x = sp.lambdify(t, x_total.subs(defaults), modules=safe_modules)
        time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
        time_y = sp.lambdify(t, y_t.subs(defaults), modules=safe_modules)
        plot_time_triplet(t, time_x, time_h, time_y, t_lim, "x(t)", "h(t)", "y(t)")

def flow_filter_time_from_freq(X_expr, t, f, defaults, symbol_map):
    safe_modules = get_safe_modules()
    h_t, H_f, from_H = resolve_filter(t, f, symbol_map)
    x_expr = sp.inverse_fourier_transform(X_expr, f, t, noconds=True)

    if from_H:
        res_str = sp.pretty(smart_simplify(x_expr, t), use_unicode=True)
        console.print(Panel(res_str, title="Risultato x(t)", border_style="green"))
        res_str = sp.pretty(smart_simplify(h_t, t), use_unicode=True)
        console.print(Panel(res_str, title="Risultato h(t)", border_style="cyan"))

    if ask_yes_no("[bold green]Mostrare grafico doppio di x(t) e h(t)? (s/n): [/bold green]"):
        t_lim_x, _ = calculate_smart_limits(x_expr, t, defaults)
        t_lim_h, _ = calculate_smart_limits(h_t, t, defaults)
        t_lim = max(t_lim_x, t_lim_h)
        time_x = sp.lambdify(t, x_expr.subs(defaults), modules=safe_modules)
        time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
        plot_time_pair(t, time_x, time_h, t_lim, "x(t)", "h(t)")

    Y_f = sp.simplify(X_expr * H_f)
    y_t = sp.inverse_fourier_transform(Y_f, f, t, noconds=True)
    res_str = sp.pretty(smart_simplify(y_t, t), use_unicode=True)
    console.print(Panel(res_str, title="Risultato y(t)", border_style="green"))
    if ask_yes_no("[bold green]Mostrare grafico triplo di x(t), h(t), y(t)? (s/n): [/bold green]"):
        t_lim_x, _ = calculate_smart_limits(x_expr, t, defaults)
        t_lim_h, _ = calculate_smart_limits(h_t, t, defaults)
        t_lim_y, _ = calculate_smart_limits(y_t, t, defaults)
        t_lim = max(t_lim_x, t_lim_h, t_lim_y)
        time_x = sp.lambdify(t, x_expr.subs(defaults), modules=safe_modules)
        time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
        time_y = sp.lambdify(t, y_t.subs(defaults), modules=safe_modules)
        plot_time_triplet(t, time_x, time_h, time_y, t_lim, "x(t)", "h(t)", "y(t)")

def print_menu(title, items):
    console.print(f"[bold cyan]{title}[/bold cyan]")
    for idx, label in enumerate(items, start=1):
        console.print(f"  {idx}) {label}")
