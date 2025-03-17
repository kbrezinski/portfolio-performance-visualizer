# Portfolio Performance Visualizer

A web-based tool built with Streamlit that allows users to compare the performance of multiple investment portfolios against a benchmark portfolio. The application fetches real-time data from Yahoo Finance and provides interactive visualizations of portfolio performance.

![Portfolio Visualizer Demo](demo_screenshot.png)

## Features

- Compare up to 4 portfolios simultaneously (1 benchmark + 3 custom portfolios)
- Real-time data fetching from Yahoo Finance
- Interactive performance visualization
- Customizable portfolio weights
- Support for multiple time periods (1 month to 5 years)
- Performance metrics with benchmark comparison
- Automatic weight normalization
- Responsive design

## Default Portfolios

### Benchmark Portfolio
- XIU.TO (35%): iShares S&P/TSX 60 Index ETF
- VFV.TO (65%): Vanguard S&P500 Index ETF
- ZAG.TO (0%): BMO Aggregate Bond Index ETF

### Custom Portfolio 1 (Default)
- AVDV (8%): Avantis International Small Cap Value ETF
- AVUV (8%): Avantis U.S. Small Cap Value ETF
- VUN.TO (25%): Vanguard U.S.Total Market Index ETF
- XEC.TO (10%): BlackRock Canada iShares Core
- XEF.TO (20%): BlackRock Canada iShares Core
- XIC.TO (29%): BlackRock iShares Core S&P/TSX Capped Composite Index ETF

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-performance-visualizer.git
cd portfolio-performance-visualizer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the interface to:
   - Modify portfolio compositions
   - Adjust time periods
   - Compare performance metrics
   - View interactive charts

## Dependencies

- Python 3.7+
- Streamlit
- yfinance
- plotly
- pandas
- numpy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data provided by Yahoo Finance through the yfinance package
- Built with Streamlit framework
- Visualization powered by Plotly

## Disclaimer

This tool is for educational and informational purposes only. It is not intended to provide investment advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. 