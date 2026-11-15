# MLBL League Analytics — Streamlit

## Files
- `app.py`: Streamlit dashboard
- `requirements.txt`: Python dependencies

## Run locally
1. Put `app.py`, `requirements.txt`, `Book1.csv`, and `Book2.csv` in the same directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

You can also upload the two CSV files in the app's sidebar, so they do not need to be beside `app.py`.

## Deploy
Push `app.py` and `requirements.txt` to a GitHub repository, then deploy the repository through Streamlit Community Cloud. Upload the CSVs through the app's sidebar after deployment, or include them in the repository if the data is intended to be public.

## Data format
Player CSV should contain:
`Number, Name, Team, G, AB, R, H, 2B, 3B, HR, RBI, BB, SO, HBP, SB, CS, ROE, AVG, OBP, SLG, OPS`

Team CSV should contain:
`Team, G, AB, R, H, 2B, 3B, HR, RBI, BB, SO, HBP, SB, CS, ROE, AVG, OBP, SLG, OPS`

The app removes blank `Unnamed: ...` columns automatically.

## OPS+ note
The app follows the notebook's simplified OPS+ formula and uses aggregate player totals to calculate league OBP and SLG. It does not apply park factors, and therefore should be described as a league-adjusted OPS+ estimate rather than official MLB OPS+.
