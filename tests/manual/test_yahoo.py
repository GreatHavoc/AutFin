import yfinance as yf

candidates = ["ETERNAL", "ETERNAL.NS", "ETERNAL.BO", "ZOMATO.NS", "ZOMATO.BO",
               "RELIANCE.NS", "RELIANCE.BO", "AAPL", "TCS.NS", "TCS.BO"]
for ticker in candidates:
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="1mo")
        last = float(h["Close"].iloc[-1]) if not h.empty else None
        rows = len(h)
        flag = "OK" if rows > 0 else "no-data"
        print(f"  {ticker:20s}  rows={rows:3d}  last={last}  [{flag}]")
    except Exception as e:
        print(f"  {ticker:20s}  ERROR  {e}")
