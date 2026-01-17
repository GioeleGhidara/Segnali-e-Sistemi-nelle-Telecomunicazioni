import sympy as sp
import numpy as np

import sst

def time_flow(x_expr, t, f, symbol_map, defaults):
    sst.console.print("[bold cyan]Flusso Tempo[/bold cyan]")
    sst.console.print("  1) Trasformata (x(t) <-> X(f))")
    sst.console.print("  2) Y(f)")
    sst.console.print("  3) y(t)")
    choice = sst.console.input("[bold green]Selezione (1-3): [/bold green]").strip()

    safe_modules = [{'sinc': sst.numeric_sinc, 'FourierTransform': sst.numeric_fallback}, 'numpy']

    if choice == "1":
        terms, base = sst.get_terms_and_base(x_expr)
        t_lim, f_lim = sst.calculate_smart_limits(base, t, defaults)
        x_total = sum(terms)
        res_str = sp.pretty(sst.smart_simplify(x_total, t), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato x(t)", border_style="green"))

        X_tot = sst.compute_total_ft(terms, t, f)
        res_str = sp.pretty(sst.smart_simplify(X_tot, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato X(f)", border_style="green"))

        time_f = sp.lambdify(t, x_total.subs(defaults), modules=safe_modules)
        freq_f = sp.lambdify(f, sp.re(X_tot).subs(defaults), modules=safe_modules)
        sst.plot_signals(t, f, time_f, freq_f, t_lim, f_lim)
        return

    if choice == "2":
        terms, base = sst.get_terms_and_base(x_expr)
        X_expr = sst.compute_total_ft(terms, t, f)
        f_lim = sst.calculate_smart_limits(base, t, defaults)[1]

        h_t, H_f, _from_H = sst.resolve_filter(t, f, symbol_map)

        res_str = sp.pretty(sst.smart_simplify(X_expr, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato X(f)", border_style="green"))
        res_str = sp.pretty(sst.smart_simplify(H_f, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato H(f)", border_style="green"))
        if sst.ask_yes_no("[bold green]Mostrare grafico di X(f) e H(f) insieme? (s/n): [/bold green]"):
            freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
            freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
            sst.plot_frequency_pair(f, freq_X, freq_H, f_lim, "X(f)", "H(f)")

        Y_f = sp.simplify(X_expr * H_f)
        res_str = sp.pretty(sst.smart_simplify(Y_f, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato Y(f)", border_style="green"))
        if sst.ask_yes_no("[bold green]Mostrare grafico di X(f), H(f), Y(f) insieme? (s/n): [/bold green]"):
            freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
            freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
            freq_Y = sp.lambdify(f, sp.re(Y_f).subs(defaults), modules=safe_modules)
            sst.plot_frequency_triplet(f, freq_X, freq_H, freq_Y, f_lim, "X(f)", "H(f)", "Y(f)")
        return

    if choice == "3":
        h_t, H_f, from_H = sst.resolve_filter(t, f, symbol_map)
        terms, _base = sst.get_terms_and_base(x_expr)
        x_total = sum(terms)

        if from_H:
            res_str = sp.pretty(sst.smart_simplify(x_total, t), use_unicode=True)
            sst.console.print(sst.Panel(res_str, title="Risultato x(t)", border_style="green"))
            res_str = sp.pretty(sst.smart_simplify(h_t, t), use_unicode=True)
            sst.console.print(sst.Panel(res_str, title="Risultato h(t)", border_style="cyan"))

        if sst.ask_yes_no("[bold green]Mostrare grafico doppio di x(t) e h(t)? (s/n): [/bold green]"):
            t_lim_x, _ = sst.calculate_smart_limits(x_total, t, defaults)
            t_lim_h, _ = sst.calculate_smart_limits(h_t, t, defaults)
            t_lim = max(t_lim_x, t_lim_h)
            time_x = sp.lambdify(t, x_total.subs(defaults), modules=safe_modules)
            time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
            sst.plot_time_pair(t, time_x, time_h, t_lim, "x(t)", "h(t)")

        X_expr = sst.compute_total_ft(terms, t, f)
        Y_f = sp.simplify(X_expr * H_f)
        y_t = sp.inverse_fourier_transform(Y_f, f, t, noconds=True)
        res_str = sp.pretty(sst.smart_simplify(y_t, t), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato y(t)", border_style="green"))
        if sst.ask_yes_no("[bold green]Mostrare grafico triplo di x(t), h(t), y(t)? (s/n): [/bold green]"):
            t_lim_x, _ = sst.calculate_smart_limits(x_total, t, defaults)
            t_lim_h, _ = sst.calculate_smart_limits(h_t, t, defaults)
            t_lim_y, _ = sst.calculate_smart_limits(y_t, t, defaults)
            t_lim = max(t_lim_x, t_lim_h, t_lim_y)
            time_x = sp.lambdify(t, x_total.subs(defaults), modules=safe_modules)
            time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
            time_y = sp.lambdify(t, y_t.subs(defaults), modules=safe_modules)
            sst.plot_time_triplet(t, time_x, time_h, time_y, t_lim, "x(t)", "h(t)", "y(t)")
        return

    sst.console.print("[bold red]Scelta non valida.[/bold red]")

def freq_flow(X_expr, t, f, symbol_map, defaults):
    sst.console.print("[bold cyan]Flusso Frequenza[/bold cyan]")
    sst.console.print("  1) Antitrasformata (X(f) -> x(t))")
    sst.console.print("  2) Filtro in frequenza")
    sst.console.print("  3) Filtro nel tempo")
    choice = sst.console.input("[bold green]Selezione (1-3): [/bold green]").strip()

    safe_modules = [{'sinc': sst.numeric_sinc, 'FourierTransform': sst.numeric_fallback}, 'numpy']

    if choice == "1":
        res_str = sp.pretty(sst.smart_simplify(X_expr, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato X(f)", border_style="green"))

        x_expr = sp.inverse_fourier_transform(X_expr, f, t, noconds=True)
        res_str = sp.pretty(sst.smart_simplify(x_expr, t), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato x(t)", border_style="green"))

        t_lim, f_lim = sst.calculate_smart_limits(x_expr, t, defaults)
        time_f = sp.lambdify(t, x_expr.subs(defaults), modules=safe_modules)
        freq_f = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
        sst.plot_signals(t, f, time_f, freq_f, t_lim, f_lim)
        return

    if choice == "2":
        h_t, H_f, _from_H = sst.resolve_filter(t, f, symbol_map)
        f_lim = sst.calculate_smart_limits(sp.exp(-t**2), t, defaults)[1]

        res_str = sp.pretty(sst.smart_simplify(X_expr, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato X(f)", border_style="green"))
        res_str = sp.pretty(sst.smart_simplify(H_f, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato H(f)", border_style="green"))
        if sst.ask_yes_no("[bold green]Mostrare grafico di X(f) e H(f) insieme? (s/n): [/bold green]"):
            freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
            freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
            sst.plot_frequency_pair(f, freq_X, freq_H, f_lim, "X(f)", "H(f)")

        Y_f = sp.simplify(X_expr * H_f)
        res_str = sp.pretty(sst.smart_simplify(Y_f, f), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato Y(f)", border_style="green"))
        if sst.ask_yes_no("[bold green]Mostrare grafico di X(f), H(f), Y(f) insieme? (s/n): [/bold green]"):
            freq_X = sp.lambdify(f, sp.re(X_expr).subs(defaults), modules=safe_modules)
            freq_H = sp.lambdify(f, sp.re(H_f).subs(defaults), modules=safe_modules)
            freq_Y = sp.lambdify(f, sp.re(Y_f).subs(defaults), modules=safe_modules)
            sst.plot_frequency_triplet(f, freq_X, freq_H, freq_Y, f_lim, "X(f)", "H(f)", "Y(f)")
        return

    if choice == "3":
        h_t, H_f, from_H = sst.resolve_filter(t, f, symbol_map)
        x_expr = sp.inverse_fourier_transform(X_expr, f, t, noconds=True)

        if from_H:
            res_str = sp.pretty(sst.smart_simplify(x_expr, t), use_unicode=True)
            sst.console.print(sst.Panel(res_str, title="Risultato x(t)", border_style="green"))
            res_str = sp.pretty(sst.smart_simplify(h_t, t), use_unicode=True)
            sst.console.print(sst.Panel(res_str, title="Risultato h(t)", border_style="cyan"))

        if sst.ask_yes_no("[bold green]Mostrare grafico doppio di x(t) e h(t)? (s/n): [/bold green]"):
            t_lim_x, _ = sst.calculate_smart_limits(x_expr, t, defaults)
            t_lim_h, _ = sst.calculate_smart_limits(h_t, t, defaults)
            t_lim = max(t_lim_x, t_lim_h)
            time_x = sp.lambdify(t, x_expr.subs(defaults), modules=safe_modules)
            time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
            sst.plot_time_pair(t, time_x, time_h, t_lim, "x(t)", "h(t)")

        Y_f = sp.simplify(X_expr * H_f)
        y_t = sp.inverse_fourier_transform(Y_f, f, t, noconds=True)
        res_str = sp.pretty(sst.smart_simplify(y_t, t), use_unicode=True)
        sst.console.print(sst.Panel(res_str, title="Risultato y(t)", border_style="green"))
        if sst.ask_yes_no("[bold green]Mostrare grafico triplo di x(t), h(t), y(t)? (s/n): [/bold green]"):
            t_lim_x, _ = sst.calculate_smart_limits(x_expr, t, defaults)
            t_lim_h, _ = sst.calculate_smart_limits(h_t, t, defaults)
            t_lim_y, _ = sst.calculate_smart_limits(y_t, t, defaults)
            t_lim = max(t_lim_x, t_lim_h, t_lim_y)
            time_x = sp.lambdify(t, x_expr.subs(defaults), modules=safe_modules)
            time_h = sp.lambdify(t, h_t.subs(defaults), modules=safe_modules)
            time_y = sp.lambdify(t, y_t.subs(defaults), modules=safe_modules)
            sst.plot_time_triplet(t, time_x, time_h, time_y, t_lim, "x(t)", "h(t)", "y(t)")
        return

    sst.console.print("[bold red]Scelta non valida.[/bold red]")

def main():
    t, f, symbol_map, defaults = sst.get_symbolic_environment()
    while True:
        sst.clear_screen()
        sst.print_user_guide()
        expr = sst.prompt_initial_expr_auto(t, f, symbol_map)
        domain = sst.resolve_domain(expr, t, f)
        if domain == "t":
            time_flow(expr, t, f, symbol_map, defaults)
        else:
            freq_flow(expr, t, f, symbol_map, defaults)

if __name__ == "__main__":
    main()
