A Machine Learning application that predicts whether a video game will be a commercial **HIT** or **FLOP** based on historical data. Built with **Random Forest** and **Streamlit**.

- **Interactive UI:** Simple web interface to input game details.
- **Smart Engineering:** Auto-calculates metrics like _Publisher Experience_ and _Competition Index_.
- **Performance:** Achieves ~74% accuracy 

## Quick Start

1. Clone the repository\*\*
   git clone [https://github.com/duongvantai16022007-cloud/game_hit_predictor.git](https://github.com/duongvantai16022007-cloud/game_hit_predictor.git)
   cd game_hit_predictor
2. Install dependencies
   pip install -r requirements.txt
3. Run the App
   streamlit run main.py
   📂 Project Structure
   main.py: Streamlit frontend interface.
   dataset.csv: Cleaned dataset (Games released after 2000).
   fix_data.py: Script for data preprocessing.
   Author: Tai Duong
