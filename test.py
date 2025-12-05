import yfinance as yf

try:
    data = yf.download("SPY ACWI", start="2020-01-01", end="2023-12-31")
    if data.empty:
        print("Aucune donnée disponible pour ces tickers et cette période")
    else:
        print(data.head())
except Exception as e:
    print(f"Erreur: {e}")
