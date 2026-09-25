# Nifty options vol-surface project
Goal: research project on NSE index options (profile/portfolio piece).
Data: ./data/date=YYYY-MM-DD/{NIFTY,BANKNIFTY}_<expiry>_<time>.parquet,
      collected live from Dhan API on a VPS, synced down via rsync.
Steps: 1 collect (done, running on VPS via cron)
       2 clean + fit SVI/SSVI per expiry  <- currently here
       3 Heston calibration + MC exotics
       4 research study  5 dashboard + write-up
Rules: Python, keep modules small, add tests, never commit .env or data/.
