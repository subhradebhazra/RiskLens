# RiskLens photo-led navigation refresh

This update changes the shared UI to a conventional horizontal top navigation and adds a photographic trading-workspace hero to the overview page.

- Main navigation is horizontal across the top; less-used pages are grouped under “More workspace pages and data source”.
- The overview hero uses a real Pexels-hosted financial workstation photo (photo page: https://www.pexels.com/photo/detailed-stock-market-analysis-workspace-36598869/).
- The image is loaded from Pexels at runtime, so an internet connection is required to display the photo. All app calculations and local demo data remain independent of the image.
- Streamlit's sidebar is hidden by the shared theme; the main page content and chart styling remain otherwise intact.

Run from the inner `RiskLens` folder:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m streamlit run app.py
```

The photo/navigation changes were syntax-compiled and the existing automated test suite passed (32 tests). A full interactive browser smoke test was not available in the build environment.
