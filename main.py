import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Knowledge Base
Cholesterol = ctrl.Antecedent(np.arange(0, 501, 1), 'Cholesterol')
BP = ctrl.Antecedent(np.arange(0, 201, 1), 'BP')
Age = ctrl.Antecedent(np.arange(0, 91, 1), 'Age')
BMI = ctrl.Antecedent(np.arange(0, 41, 1), 'BMI')
Diabetes = ctrl.Antecedent(np.arange(0, 401, 1), 'Diabetes')
Health = ctrl.Consequent(np.arange(0, 5, 1), 'Health')

Cholesterol['Normal'] = fuzz.trapmf(Cholesterol.universe, [0, 0, 50, 200])
Cholesterol['Medium'] = fuzz.trimf(Cholesterol.universe, [190, 215, 239])
Cholesterol['High'] = fuzz.trimf(Cholesterol.universe, [240, 265, 320])
Cholesterol['Very high'] = fuzz.trapmf(Cholesterol.universe, [280, 365, 500, 500])

BP['Low'] = fuzz.trapmf(BP.universe, [0, 0, 75, 90])
BP['Medium'] = fuzz.trimf(BP.universe, [90, 100, 120])
BP['High'] = fuzz.trapmf(BP.universe, [120, 163, 200, 200])

Age['Young'] = fuzz.trapmf(Age.universe, [0, 0, 21, 40])
Age['Middle-age'] = fuzz.trimf(Age.universe, [40, 49, 59])
Age['Old'] = fuzz.trimf(Age.universe, [59, 69, 70])
Age['Very old'] = fuzz.trapmf(Age.universe, [70, 79, 89, 89])
Age.view()

BMI['Under-weight'] = fuzz.trapmf(BMI.universe, [0, 0, 11, 18.5])
BMI['Normal'] = fuzz.trimf(BMI.universe, [18.5, 20, 24.9])
BMI['Over-weight'] = fuzz.trimf(BMI.universe, [24, 27, 30])
BMI['Obese'] = fuzz.trapmf(BMI.universe, [30, 35, 40, 40])


Diabetes['Normal'] = fuzz.trimf(Diabetes.universe, [0, 70, 199])
Diabetes['Diabetic'] = fuzz.trimf(Diabetes.universe, [199, 230, 400])

Health['Healthy'] = fuzz.trapmf(Health.universe, [0, 0, 0.3, 1.7])
Health['Early stage'] = fuzz.trimf(Health.universe, [1.5, 2, 2.5])
Health['Advanced stage'] = fuzz.trapmf(Health.universe, [2.4, 3, 4, 4])

# Rule Base
rule1 = ctrl.Rule(Age['Middle-age'] & BP['Medium'] & Cholesterol['Normal'] & Diabetes['Normal'] & BMI['Normal'],
                  Health['Healthy'])
rule2 = ctrl.Rule(Age['Old'] & BP['Medium'] & Cholesterol['Medium'] & Diabetes['Normal'] & BMI['Over-weight'],
                  Health['Healthy'])
rule3 = ctrl.Rule(Age['Young'] & BP['Medium'] & Cholesterol['Medium'] & Diabetes['Normal'] & BMI['Over-weight'],
                  Health['Healthy'])
rule4 = ctrl.Rule(Age['Old'] & BP['High'] & Cholesterol['Very high'] & Diabetes['Diabetic'] & BMI['Obese'],
                  Health['Early stage'])
rule5 = ctrl.Rule(Age['Very old'] & BP['High'] & Cholesterol['High'] & Diabetes['Normal'] & BMI['Obese'],
                  Health['Early stage'])
rule6 = ctrl.Rule(Age['Very old'] & BP['High'] & Cholesterol['Very high'] & Diabetes['Diabetic'] & BMI['Obese'],
                  Health['Advanced stage'])

Health_Sys = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
Health1 = ctrl.ControlSystemSimulation(Health_Sys)

Health1.input['Age'] = 79
Health1.input['BP'] = 163
Health1.input['Cholesterol'] = 365
Health1.input['Diabetes'] = 230
Health1.input['BMI'] = 35

Health1.compute()
print(Health1.output['Health'])
Health.view(sim=Health1)
