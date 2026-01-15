# Spettrale - Segnali e Sistemi

Strumento didattico per l'analisi di segnali nel tempo e nel dominio della frequenza con SymPy, NumPy e Matplotlib.

## Requisiti
- Python 3.8+
- sympy
- numpy
- matplotlib
- rich

## Avvio
```powershell
python sst.py
```

## Esempi di input
- `rect(t)`
- `tri(t)`
- `u(t) - u(t-1)`
- `rep(rect(t), 4)`
- `conv(rect(t), rect(t))`

## Note
- Usa `*` per moltiplicare, ad esempio `2*T`.
- Il grafico in frequenza mostra la parte reale di `X(f)`.
