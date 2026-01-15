import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# --- CONFIGURAZIONE ---
N_REP = 4  # Numero di repliche per lato

console = Console()

# --- FUNZIONI NUMERICHE ROBUSTE ---
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

# --- LEGENDA COMPLETA RIPRISTINATA ---
def print_user_guide():
    console.print()
    console.rule("[bold cyan]ANALIZZATORE SPETTRALE (FULL EDITION)[/bold cyan]")
    
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
    
    console.print(table)
    console.print("[yellow]Nota:[/yellow] Usa sempre l'asterisco per moltiplicare (es. [bold]2*T[/bold]).\n")

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

    symbol_map = {
        't': t, 'f': f,
        'rect': piecewise_rect, 'tri': piecewise_tri, 'u': sp.Heaviside,
        'sinc': sp.sinc, 'pi': sp.pi, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 
        'abs': sp.Abs, 'j': sp.I, 'DiracDelta': sp.DiracDelta,
        'rep': gen_rep, 'comb': gen_comb, 'conv': gen_conv,
        'T': T, 'A': A, 'f0': f0, 'T0': T0
    }
    default_values = {T: 1, T0: 1, A: 1, f0: 1}
    return t, f, symbol_map, default_values

# --- PRETTY PRINTING ---
def make_readable(expr):
    visual_exp = sp.Function('exp')
    expr_clean = expr.replace(sp.exp, visual_exp)
    visual_sinc = sp.Function('sinc')
    expr_clean = expr_clean.replace(sp.sinc, lambda arg: visual_sinc(sp.simplify(arg / sp.pi)))
    return expr_clean

def smart_simplify(expr, f_symbol):
    if isinstance(expr, sp.Piecewise):
        for e, c in expr.args:
            if not e.is_constant(): expr = e; break
            
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

# --- AUTO ZOOM ---
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

def run_analysis():
    t, f, symbol_map, defaults = get_symbolic_environment()
    print_user_guide()

    raw_input = console.input("[bold green]Inserisci x(t): [/bold green]")
    
    try:
        obj = eval(raw_input.replace('^', '**'), symbol_map)
        
        # Limiti Grafico
        if isinstance(obj, PeriodicSignal): target = obj.base_signal
        elif isinstance(obj, list): target = sum(obj)
        else: target = obj
        t_lim, f_lim = calculate_smart_limits(target, t, defaults)

        # Analisi Simbolica
        if isinstance(obj, PeriodicSignal):
            console.print(Panel("[yellow]Segnale Periodico Rilevato[/yellow]"))
            # Usa smart_simplify per formule pulite
            X_base = smart_simplify(sp.fourier_transform(obj.base_signal, t, f, noconds=True), f)
            M = 2 * N_REP + 1; T0_val = obj.period
            dirichlet = sp.sin(M * sp.pi * f * T0_val) / sp.sin(sp.pi * f * T0_val)
            
            console.print(Panel(sp.pretty(X_base, use_unicode=True), title="Inviluppo (Base)", border_style="cyan"))
            console.print(Panel(sp.pretty(dirichlet, use_unicode=True), title="Periodicità", border_style="magenta"))
            terms = obj.terms
        else:
            if isinstance(obj, list): terms = obj
            else: terms = [obj]
            
        # Calcolo Numerico
        x_total = sum(terms)
        safe_modules = [{'sinc': numeric_sinc, 'FourierTransform': numeric_fallback}, 'numpy']
        
        time_f = sp.lambdify(t, x_total.subs(defaults), modules=safe_modules)
        
        X_tot = 0
        with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.percentage:>3.0f}%")) as p:
            task = p.add_task("Calcolo Trasformata...", total=len(terms))
            for term in terms:
                res = sp.fourier_transform(term, t, f, noconds=True)
                X_tot += res
                p.advance(task)
        
        freq_f = sp.lambdify(f, sp.re(X_tot).subs(defaults), modules=safe_modules)
        
        if not isinstance(obj, PeriodicSignal):
            res_str = sp.pretty(smart_simplify(X_tot, f), use_unicode=True)
            console.print(Panel(res_str, title="Risultato X(f)", border_style="green"))

        plot_signals(t, f, time_f, freq_f, t_lim, f_lim)

    except Exception as e:
        console.print(f"[bold red]ERRORE:[/bold red] {e}")

if __name__ == "__main__":
    run_analysis()

