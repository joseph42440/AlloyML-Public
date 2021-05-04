import pickle

models = {"DoS": pickle.load(open("models/DoS_Model_1619609637.sav", "rb")),
          "elongation": pickle.load(open("models/Elongation_Model_1619609639.sav", "rb")),
          "tensile": pickle.load(open("models/Tensile_Strength_Model_1619609643.sav", "rb")),
          "yield": pickle.load(open("models/Yield_Strength_Model_1619609646.sav", "rb")),
          }

