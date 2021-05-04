from copy import deepcopy
import numpy as np
from scipy.stats import truncnorm
import pickle
if 'google.colab' in str(get_ipython()):
    from AlloyML_Public.model_paths import models
else:
    from model_paths import models

class AlDatapoint:
    def __init__(self, settings):
        self.constant_inputs = settings.constant_inputs
        self.categorical_inputs = settings.categorical_inputs
        self.categorical_inputs_info = settings.categorical_inputs_info
        self.range_based_inputs = settings.range_based_inputs

    def formatForInput(self):
        my_input = [*self.constant_inputs.values(), *self.categorical_inputs.values(), self.getAl(),
                    *self.range_based_inputs.values()]
        return np.reshape(my_input, (1, -1))

    def print(self):
        for key, value in self.constant_inputs.items():
            print(f"{key}: {value}")
        for key, value in self.categorical_inputs.items():
            print(f"{key}: {self.categorical_inputs_info[key]['tag'][self.categorical_inputs_info[key]['span'].index(value)]}")
        print(f"Al%: {round(self.getAl(), 2)}")
        for key, value in self.range_based_inputs.items():
            if value:
                print(f"{key}: {value}")

    def getAl(self):
        return 100 - sum(self.range_based_inputs.values())


class scanSettings:
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'DoS':
            self.loss_type = 'Linear'
            self.max_steps = 10
            self.targets = {
                'DoS': 10
            }
            self.constant_inputs = {
                'time(days)': 7,
                'temperature(C)': 150,
            }
            self.categorical_inputs = {
                'recrystallised': [1],
                'temper': [1, 2, 4, 5]
            }
            self.categorical_inputs_info = {
                'recrystallised': {'span': [0, 1], 'tag': ['No', 'Yes']},
                'temper': {'span': [1, 2, 3, 4, 5, 6, 7], 'tag': ['H (lab)', 'H116', 'H131', 'H321', "O",
                                                                  "Stabilised", "Unknown"]}}

            self.range_based_inputs = dict.fromkeys(
                ['Ag%', 'Ca%', 'Ce%', 'Cr%', 'Cu%', 'Er%', 'Fe%', 'Ge%', 'Mg%', 'Mn%',
                 'Nd%', 'Ni%', 'Sc%', 'Si%', 'Sr%', 'Ti%', 'Zn%', 'Zr%'], [0, 0])
            self.range_based_inputs['Mg%'] = [4.0, 5.5]

        if self.mode == 'Mechanical':
            self.loss_type = 'Percentage'
            self.max_steps = 10
            self.targets = {
                'elongation%': 6,
                'tensile strength(MPa)': 250
            }
            self.constant_inputs = {}
            self.categorical_inputs = {
                'processing condition': [1, 2, 4, 6, 9, 10],
            }
            self.categorical_inputs_info = {
                'processing condition': {'span': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                         'tag': ['As-cast or as-fabricated', 'Annealed, solutionised',
                                                 'H (soft)', 'H (hard)', 'T1', 'T2', 'T3', 'T4', 'T5',
                                                 'T6', 'T7', 'T8', 'Lab routine or unknown']},
            }
            self.range_based_inputs = dict.fromkeys(
                ['Ag%', 'B%', 'Be%', 'Bi%', 'Cd%', 'Co%', 'Cr%', 'Cu%', 'Er%', 'Eu%', 'Fe%',
                 'Ga%', 'Li%', 'Mg%', 'Mn%', 'Ni%', 'Pb%', 'Sc%', 'Si%', 'Sn%', 'Ti%', 'V%',
                 'Zn%', 'Zr%'], [0, 0])
            self.range_based_inputs['Mg%'] = [4.0, 5.5]


class optimiser:
    def __init__(self, settings):
        self.step_batch_size = 100
        self.step_final_std = 0.01
        self.finetune_max_rounds = 3
        self.finetune_batch_size = 10
        self.mode = settings.mode
        self.loss_type = settings.loss_type
        self.targets = settings.targets
        self.max_steps = settings.max_steps
        self.constant_inputs = settings.constant_inputs
        self.categorical_inputs = settings.categorical_inputs
        self.range_based_inputs = settings.range_based_inputs
        self.settings = settings
        self.models = models

        self.run()

    def calculateLoss(self, datapoint):
        if self.mode == 'DoS':
            return abs(self.models['DoS'].predict(datapoint.formatForInput())[0] - self.targets['DoS'])
        elif self.mode == 'Mechanical':
            return ((abs((self.models['elongation'].predict(datapoint.formatForInput())[0] / self.targets[
                'elongation%']) - 1) * 100
                     + abs((self.models['tensile'].predict(datapoint.formatForInput())[0] / self.targets[
                        'tensile strength(MPa)']) - 1) * 100)) / 2

    def printResults(self, best_datapoint):
        best_datapoint.print()
        if self.mode == 'DoS':
            print('Results in a predicted %f DoS' % (self.models['DoS'].predict(best_datapoint.formatForInput())[0]))
        elif self.mode == 'Mechanical':
            print('Results in a predicted %f elongation(%%)' % (
            self.models['elongation'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f tensile strength(MPa)' % (
            self.models['tensile'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f yield strength(MPa)' % (
            self.models['yield'].predict(best_datapoint.formatForInput())[0]))

    def run(self):
        best_loss = None
        best_datapoint = AlDatapoint(self.settings)
        for i in range(self.max_steps):
            loss, datapoint = self.calculateStep(best_datapoint, i, 'all')
            if best_loss is None or loss < best_loss:
                best_datapoint = datapoint
                best_loss = loss
            print('[Step %d/%d] Best %s Loss = %f.' % (i+1, self.max_steps, self.loss_type, best_loss))

        for i in range(self.finetune_max_rounds):
            for key in [*self.categorical_inputs.keys(), *self.range_based_inputs.keys()]:
                loss, datapoint = self.calculateStep(best_datapoint, i, key)
                if loss < best_loss:
                    best_datapoint = datapoint
                    best_loss = loss
                print('[Finetune %s] Best %s Loss = %f.' % (key, self.loss_type, best_loss))
            else:
                break
        print('==========Scan Finished==========')
        self.printResults(best_datapoint)

    def calculateStep(self, best_datapoint, step_number, target_var):
        if target_var == 'all':
            batch_size = self.step_batch_size
        else:
            batch_size = self.finetune_batch_size
        loss = [0] * batch_size
        datapoints = []
        std = self.step_final_std * (self.max_steps / float(step_number + 1))
        for i in range(batch_size):
            datapoints.append(deepcopy(best_datapoint))
            for key in self.categorical_inputs.keys():
                if target_var == key or target_var == 'all':
                    datapoints[i].categorical_inputs[key] = np.random.choice(self.categorical_inputs[key])
            for key in self.range_based_inputs.keys():
                if target_var == key or target_var == 'all':
                    if max(self.range_based_inputs[key]) != min(self.range_based_inputs[key]):
                        a = (min(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        b = (max(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        datapoints[i].range_based_inputs[key] = round(
                            float(truncnorm.rvs(a, b, loc=np.mean(best_datapoint.range_based_inputs[key]), scale=std)),
                            2)
                    else:
                        datapoints[i].range_based_inputs[key] = min(self.range_based_inputs[key])
            loss[i] = self.calculateLoss(datapoints[i])
        return min(loss), datapoints[loss.index(min(loss))]