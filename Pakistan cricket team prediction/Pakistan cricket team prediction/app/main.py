import sys
from PyQt5.QtWidgets import QApplication
from match_predictor import MatchPredictorModel
from match_ui import MatchPredictorApp

def main():
    app = QApplication(sys.argv)
    
    predictor_logic = MatchPredictorModel("data/cric.csv")
    
    window = MatchPredictorApp(predictor_logic)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()