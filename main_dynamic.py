import sympy as sp
import numpy as np

import sst

def run_flow(expr, domain, t, f, symbol_map, defaults):
    if domain == "t":
        sst.print_menu("Flusso Tempo", [
            "Trasformata/Antitrasformata (auto)",
            "Filtro in frequenza",
            "Filtro nel tempo",
        ])
        choice = sst.console.input("[bold green]Selezione (1-3): [/bold green]").strip()

        if choice == "1":
            sst.flow_transform_auto(expr, t, f, defaults)
            return

        if choice == "2":
            sst.flow_filter_frequency_from_time(expr, t, f, defaults, symbol_map)
            return

        if choice == "3":
            sst.flow_filter_time_from_time(expr, t, f, defaults, symbol_map)
            return

        sst.console.print("[bold red]Scelta non valida.[/bold red]")
        return

    sst.print_menu("Flusso Frequenza", [
        "Trasformata/Antitrasformata (auto)",
        "Filtro in frequenza",
        "Filtro nel tempo",
    ])
    choice = sst.console.input("[bold green]Selezione (1-3): [/bold green]").strip()

    if choice == "1":
        sst.flow_transform_auto(expr, t, f, defaults)
        return

    if choice == "2":
        sst.flow_filter_frequency_from_freq(expr, t, f, defaults, symbol_map)
        return

    if choice == "3":
        sst.flow_filter_time_from_freq(expr, t, f, defaults, symbol_map)
        return

    sst.console.print("[bold red]Scelta non valida.[/bold red]")

def main():
    t, f, symbol_map, defaults = sst.get_symbolic_environment()
    while True:
        sst.clear_screen()
        sst.print_user_guide()
        expr = sst.prompt_initial_expr_auto(t, f, symbol_map)
        domain = sst.resolve_domain(expr, t, f)
        run_flow(expr, domain, t, f, symbol_map, defaults)

if __name__ == "__main__":
    main()
