import csv
import os

bolus = []
cgm = []

data_path = os.path.abspath('../data')

with open(os.path.join(data_path, 'BolusData.csv')) as fp:
    reader = csv.reader(fp)
    for row in reader:
        bolus.append(row)


with open(os.path.join(data_path, 'CGMData.csv')) as fp:
    reader = csv.reader(fp)
    for row in reader:
        cgm.append(row)

count = 0
cgm_final = []
meal_final = []
cgm[0] = [float(i) for i in cgm[0]]
cgm[1] = [float(i) for i in cgm[1]]
bolus[0] = [float(i) for i in bolus[0]]
bolus[1] = [float(i) for i in bolus[1]]
cgm[0].reverse()
cgm[1].reverse()
bolus[0].reverse()
bolus[1].reverse()
row1 = cgm[0]
row2 = cgm[1]
gt_date = bolus[0]
gt = bolus[1]
prev_end = 0
for num in range(len(gt)):
    if num >= len(cgm[0])-1 or float(row1[num]) != float(row1[num+1]):
        cgm_final.append(row2[prev_end:num+1])
        meal_final.append([i for i, val in enumerate(gt[prev_end:num+1]) if float(val) > 1])
        prev_end = num + 1


with open(os.path.join(data_path, 'CGM_Regression_DayWise.csv'), 'w') as fp:
    writer = csv.writer(fp)
    for row in cgm_final:
        writer.writerow(row)

with open(os.path.join(data_path, 'Ground_Truth_DayWise.csv'), 'w') as fp:
    writer = csv.writer(fp)
    for row in meal_final:
        writer.writerow(row)
