//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/WeightHacks.h"
#include "n3rd/FullyConnectedLayer.h"
#include "n3rd/FullyConnectedLayerBlas.h"
#include "n3rd/TemporalConvolutionalLayer.h"
#include "n3rd/TemporalConvolutionalLayerBlas.h"

using namespace sgdtk;
using namespace n3rd;

const static double WEIGHTS[] = { 0.55378, 0.85794, 0.28704, 0.77357, 0.60346, 0.66414, 0.78869, 0.57215, 0.19649, 0.55486, 0.047278, 0.91051, 0.93499, 0.48493, 0.66972, 0.73655, 0.81481, 0.69474, 0.030946, 0.76521, 0.17243, 0.67755, 0.073254, 0.70875, 0.27089, 0.78897, 0.49073, 0.48164, 0.51919, 0.6046, 0.61136, 0.37237, 0.82681, 0.016078, 0.56538, 0.24744, 0.5025, 0.25193, 0.27155, 0.69264, 0.48626, 0.15963, 0.44172, 0.26304, 0.54127, 0.2704, 0.89558, 0.80333, 0.69617, 0.47597, 0.65808, 0.05224, 0.20711, 0.047844, 0.6096, 0.34538, 0.58784, 0.81613, 0.69641, 0.95468, 0.73267, 0.95176, 0.75042, 0.22127, 0.04797, 0.92625, 0.48976, 0.68381, 0.058479, 0.48232, 0.11956, 0.56521, 0.35605, 0.64432, 0.55882, 0.23183, 0.14748, 0.8815, 0.71617, 0.60561, 0.19025, 0.41429, 0.64219, 0.45183, 0.30578, 0.21738, 0.96686, 0.31283, 0.3263, 0.5763, 0.62077, 0.33815, 0.95117, 0.30315, 0.76121, 0.32326, 0.32112, 0.10606, 0.26597,  0.675, 0.23109, 0.79857, 0.34044, 0.041833, 0.049715, 0.32671, 0.0062986, 0.7665, 0.69564, 0.14311, 0.14742, 0.57925, 0.54011, 0.095794, 0.024575, 0.1508,  0.546, 0.15901, 0.13457, 0.98632, 0.63243, 0.27683, 0.40013, 0.64601, 0.78192, 0.83562, 0.90759, 0.52572, 0.030169, 0.88981, 0.05905, 0.17441,  0.794, 0.13328, 0.50239, 0.048407, 0.9192, 0.28806, 0.093842, 0.61034, 0.059089, 0.25254, 0.40043, 0.44685, 0.33553, 0.46804, 0.13383, 0.69497, 0.37456, 0.50645, 0.39676, 0.14442, 0.91142, 0.2289, 0.49334, 0.78265, 0.41619, 0.33515, 0.93399, 0.22974, 0.41178, 0.65613, 0.62835, 0.54298, 0.24517, 0.83738, 0.28193, 0.0088164, 0.051956, 0.67658, 0.056091, 0.38936, 0.92554, 0.15761, 0.10477, 0.24679, 0.66159, 0.15103, 0.49571, 0.32398, 0.92794, 0.73142, 0.096512, 0.75753, 0.49084, 0.53149, 0.57066, 0.86888, 0.85466, 0.47729, 0.42885, 0.36911, 0.68342, 0.53217, 0.74319, 0.82032, 0.34094, 0.74509, 0.96005, 0.31779, 0.62029, 0.73775, 0.90718, 0.58026, 0.17101, 0.60034, 0.98956, 0.0052629, 0.86138, 0.56905, 0.010618, 0.032155, 0.66358, 0.48091, 0.60453, 0.66451, 0.34518, 0.96184, 0.087089, 0.41762, 0.34329, 0.33285, 0.98048, 0.53999, 0.95254, 0.29981, 0.87081, 0.84737, 0.93165, 0.077467, 0.37483, 0.2086, 0.66327, 0.17648, 0.53913, 0.86508, 0.23279,  0.493, 0.99098, 0.9311, 0.14757, 0.71066, 0.37353, 0.65212, 0.28478, 0.61801, 0.38539, 0.57312, 0.7051, 0.075665, 0.0021521, 0.11006, 0.87373, 0.55939, 0.67832, 0.78822, 0.62364, 0.67756, 0.66819, 0.29938, 0.078509, 0.4977, 0.28706, 0.28632, 0.79359, 0.7304, 0.53973, 0.27815, 0.2341, 0.79274, 0.23414, 0.075727, 0.82638, 0.1238, 0.79356, 0.72403,  0.581, 0.27085, 0.96762, 0.089287, 0.76576, 0.29352, 0.05756, 0.63665, 0.78047, 0.51814, 0.95042, 0.38417, 0.26975, 0.81812, 0.072896, 0.69147, 0.62205, 0.50491, 0.50851, 0.68794, 0.22048, 0.16991, 0.80646, 0.96882, 0.71217, 0.11232, 0.024277, 0.84679, 0.73378, 0.54139, 0.45172, 0.95929, 0.89025, 0.61506, 0.1052, 0.85733, 0.91068, 0.3892, 0.10395, 0.048947, 0.58307, 0.10622, 0.48773, 0.72274, 0.3515, 0.16368, 0.064516, 0.85652, 0.79787, 0.5271, 0.92584, 0.0062699, 0.15966, 0.27275, 0.51861, 0.090467, 0.30318, 0.73496, 0.66074, 0.17731, 0.068176, 0.91745, 0.11417, 0.22945, 0.041248, 0.98229, 0.86773, 0.87302, 0.22826, 0.079771, 0.056004, 0.80387, 0.34156, 0.98011, 0.61097, 0.88747, 0.34154, 0.72582, 0.63326, 0.61204, 0.59974, 0.47043, 0.41144, 0.59249, 0.63188, 0.54701, 0.17064, 0.48744, 0.83755, 0.7783, 0.44227, 0.2782, 0.61478, 0.43796, 0.71866, 0.017508, 0.69448, 0.14091, 0.75674,  0.907, 0.66877, 0.76194, 0.65768, 0.84723, 0.46652, 0.40769, 0.43305, 0.7163, 0.60501, 0.027998, 0.96647, 0.16196, 0.18699, 0.53952, 0.75292, 0.68469, 0.2231, 0.12321, 0.42143, 0.57657, 0.96227, 0.81405, 0.24927, 0.22608, 0.35341, 0.33998, 0.68006, 0.58718, 0.035439, 0.081833, 0.17843, 0.65689, 0.041424, 0.13185, 0.1284, 0.52469, 0.64636, 0.20391, 0.55027, 0.85278, 0.67581, 0.85772, 0.51308, 0.072346, 0.86101, 0.32567, 0.032696, 0.73761, 0.11459, 0.20966, 0.59523, 0.46853, 0.01652, 0.61992, 0.79128, 0.38644, 0.51006, 0.86544, 0.54919, 0.017212, 0.43873, 0.51339, 0.89221, 0.86857, 0.56644, 0.6614, 0.083183, 0.067423, 0.73498, 0.21087, 0.28082, 0.31155, 0.3855, 0.018554, 0.12704, 0.89302, 0.82277, 0.68735, 0.42509, 0.22835, 0.37558, 0.31332, 0.04621, 0.68341, 0.04308, 0.070607, 0.73446, 0.38663, 0.4192, 0.98063, 0.028704, 0.36261, 0.48664, 0.86108, 0.77766, 0.85031, 0.73443, 0.26517, 0.69017, 0.22173, 0.24276, 0.33249, 0.077998, 0.070966, 0.53663, 0.4448, 0.61237, 0.057609, 0.24335, 0.31142, 0.30237, 0.42065, 0.34109, 0.87918, 0.42134, 0.10981, 0.96836, 0.95884, 0.9881, 0.60032, 0.23893, 0.82707, 0.078411, 0.15723, 0.3143, 0.98704, 0.86276, 0.74036, 0.034773, 0.83817, 0.46847, 0.62868, 0.37007, 0.25787, 0.088229, 0.24017, 0.61357, 0.37159, 0.91805, 0.05577, 0.15989, 0.28588, 0.58263, 0.28022, 0.25919, 0.32032, 0.81497, 0.98614, 0.12301, 0.72065, 0.31212, 0.65396, 0.63141, 0.85326, 0.14131, 0.39516, 0.97431, 0.78924, 0.97275, 0.27064, 0.84357, 0.31495, 0.38556, 0.87144, 0.96031, 0.26988, 0.68884, 0.8312, 0.38007, 0.018602, 0.66384, 0.33014, 0.45038, 0.095517, 0.5663, 0.41848, 0.7802, 0.4516, 0.89339, 0.029504, 0.60174, 0.54317, 0.036685, 0.91639, 0.44942, 0.39956, 0.86803, 0.14118, 0.98448, 0.7305, 0.95176, 0.88095, 0.050216, 0.82969, 0.00078123, 0.22375, 0.83001, 0.34894, 0.23745, 0.99184, 0.51835, 0.60463, 0.15139, 0.44246, 0.70618, 0.89009, 0.73407, 0.21273, 0.040361, 0.66979, 0.10299, 0.44033, 0.5984, 0.92984, 0.43054, 0.045028, 0.67087, 0.67796, 0.98532, 0.45873, 0.49301, 0.53116, 0.93922, 0.5942, 0.54106, 0.50494, 0.23433, 0.0037753, 0.95421, 0.61446, 0.023163, 0.89903, 0.86762, 0.32877, 0.96298, 0.31054, 0.75961, 0.7984, 0.26189, 0.80364, 0.84142, 0.64747, 0.18228, 0.82206, 0.57169, 0.7017, 0.59359, 0.49963, 0.34792, 0.46755, 0.084094, 0.64291, 0.10814, 0.8161, 0.015853,  0.491, 0.87853, 0.96571, 0.72767, 0.72236, 0.2344, 0.87233, 0.4365, 0.28687, 0.32717, 0.07865, 0.15042, 0.27648, 0.050633, 0.24916, 0.17462, 0.24857, 0.47635, 0.36155, 0.79973, 0.63933, 0.09299, 0.87299, 0.99377, 0.91393, 0.81339, 0.65029, 0.95919, 0.2541, 0.073643, 0.08485, 0.30294, 0.89878, 0.073168, 0.82964, 0.69686, 0.67174, 0.85607, 0.87911, 0.76347, 0.87168, 0.45974, 0.18732, 0.14186, 0.97428, 0.14072, 0.60331, 0.13513, 0.082466, 0.46849, 0.20409, 0.11333, 0.6363, 0.42042, 0.27268, 0.24344, 0.62678, 0.78784, 0.071369, 0.31799, 0.16595, 0.35324, 0.16787, 0.073845, 0.66203, 0.59316, 0.15285, 0.59787, 0.7553, 0.16755, 0.35624, 0.01149, 0.14073, 0.43183, 0.38785, 0.23455, 0.049499, 0.67038, 0.063357, 0.066612, 0.25189, 0.51301, 0.70433, 0.75954, 0.87818, 0.015987, 0.81863, 0.44468, 0.0075508, 0.25319, 0.78083, 0.86664, 0.67433, 0.41263, 0.32975, 0.45177, 0.74611, 0.46023, 0.54477, 0.99729, 0.39888, 0.97886, 0.73172, 0.82532, 0.83906, 0.82664, 0.86136, 0.50038, 0.49414, 0.82133,   0.19, 0.81338, 0.94505, 0.32208, 0.72325, 0.2979, 0.87789, 0.097488, 0.4587, 0.55655, 0.22128, 0.31864, 0.21502, 0.90406, 0.85541, 0.13048, 0.75715, 0.66402, 0.64374, 0.3241, 0.17721, 0.039926, 0.69496, 0.49775, 0.69527, 0.44164, 0.16852, 0.44493, 0.7915, 0.35171, 0.30816, 0.72421, 0.43032, 0.41657, 0.59348, 0.68779, 0.050732, 0.15889, 0.85157, 0.095414, 0.055109, 0.23391, 0.74403, 0.27825, 0.14202, 0.84275, 0.84924, 0.57555, 0.53597, 0.84892, 0.84953, 0.46466, 0.39771, 0.57547, 0.7729, 0.19016, 0.40022, 0.57244, 0.49577, 0.87763, 0.52019, 0.07727, 0.60298, 0.82884, 0.52185, 0.30129, 0.75849, 0.78341, 0.76049, 0.18575, 0.76007, 0.14004, 0.33416, 0.89256, 0.78157, 0.39034, 0.66941, 0.87773, 0.28403, 0.7405, 0.37476, 0.67481, 0.76062, 0.85341, 0.16196, 0.19201, 0.35601, 0.40558, 0.038837, 0.72911, 0.18412, 0.25976, 0.26245, 0.31785, 0.20333, 0.98113, 0.062243, 0.46537, 0.6049, 0.66833, 0.082985, 0.56904, 0.34165, 0.5755, 0.70537, 0.4435, 0.92175, 0.93303, 0.65508, 0.61299, 0.073409, 0.17457, 0.99875, 0.21815, 0.84138, 0.74169, 0.55656, 0.65401, 0.40871, 0.15346, 0.88037, 0.034577, 0.25883, 0.95235, 0.1981, 0.12711, 0.39662, 0.62488, 0.41924, 0.28113, 0.31432, 0.19758, 0.28869, 0.20242, 0.2376, 0.26644, 0.65542, 0.83101, 0.45718, 0.40028, 0.73281, 0.84801, 0.34588, 0.13449, 0.29257, 0.10771, 0.80793, 0.94067, 0.04131, 0.63444, 0.84547, 0.99115, 0.42422, 0.86509, 0.070281, 0.11778, 0.59432, 0.8921, 0.83917, 0.90177, 0.13758, 0.66834, 0.97642, 0.087686, 0.74838, 0.57639, 0.95179, 0.79642, 0.67813, 0.23194, 0.17476, 0.82209, 0.69587, 0.49535, 0.21595, 0.014843, 0.5523, 0.15988, 0.41009, 0.082094, 0.33597, 0.85305, 0.52385, 0.78744, 0.16756, 0.75748, 0.79412, 0.62475, 0.56466, 0.62478, 0.55778, 0.22327, 0.15662, 0.024053, 0.23069, 0.83056, 0.89597, 0.73113, 0.51693, 0.95244, 0.26182, 0.25467, 0.10302, 0.5805, 0.15114, 0.56612, 0.87969, 0.77504, 0.36431, 0.022235, 0.75337, 0.83669, 0.43299, 0.34832, 0.8519, 0.67609, 0.38361, 0.64866, 0.75276, 0.89275, 0.55599, 0.49713, 0.73495, 0.75992, 0.83052, 0.40595, 0.46281, 0.45493, 0.62478, 0.98673, 0.068462, 0.16653, 0.95009, 0.11221, 0.47304, 0.70182, 0.64555, 0.51874, 0.99264, 0.1936, 0.3906, 0.59249, 0.10845, 0.043771, 0.99754, 0.058597, 0.89677, 0.75208, 0.99914, 0.062349, 0.4946, 0.56944, 0.95969, 0.6592, 0.20251, 0.17199, 0.21301, 0.00079535, 0.4226, 0.60848, 0.84996, 0.35201, 0.47759, 0.014161, 0.87195, 0.44177, 0.060211, 0.89852, 0.20283, 0.53661, 0.11202, 0.58346, 0.29958, 0.85459, 0.40284, 0.28054, 0.32548, 0.92493, 0.81807, 0.19836, 0.55683, 0.31932, 0.28894, 0.80113, 0.29107, 0.14444, 0.48715, 0.20087, 0.97584, 0.36331, 0.092647, 0.88134, 0.89309, 0.64581, 0.075218, 0.35383, 0.29076, 0.39468, 0.39452, 0.86817, 0.3707, 0.12515, 0.096214, 0.17927, 0.90034, 0.0043511, 0.64798, 0.059648, 0.46347, 0.14007, 0.48681, 0.72708, 0.9129, 0.13626, 0.81193, 0.96261, 0.77559, 0.44156, 0.00995, 0.95876, 0.17011, 0.85869, 0.49976, 0.24484, 0.35153, 0.37674, 0.29188, 0.84587, 0.043989, 0.6647, 0.67157, 0.38581, 0.1148, 0.47158, 0.23521, 0.93742, 0.77314, 0.47121, 0.5415, 0.2055, 0.50828, 0.44315, 0.24206, 0.92093, 0.23266, 0.098751, 0.38565, 0.86426, 0.42714, 0.72474, 0.24294, 0.25481, 0.69882, 0.9234, 0.4071, 0.56875, 0.046591, 0.50931, 0.062237, 0.66972, 0.60402, 0.83993, 0.57278, 0.41587, 0.50129, 0.43668, 0.93945, 0.21583, 0.94214, 0.91786, 0.25664, 0.12908, 0.35358, 0.0041818,  0.614, 0.77559, 0.56057, 0.030412, 0.73057, 0.021461, 0.81447,  0.734, 0.71509, 0.58165, 0.67944, 0.68684, 0.96932, 0.096094, 0.37593, 0.14346, 0.89152, 0.61709, 0.094217, 0.89144, 0.9137, 0.058088, 0.95512, 0.26405, 0.25035, 0.035282, 0.89059, 0.50842, 0.42795, 0.94747, 0.10487, 0.7272, 0.79384, 0.8022, 0.53503, 0.9522, 0.72345, 0.58947, 0.90697, 0.73816, 0.93836, 0.38151, 0.33862, 0.55937, 0.94331, 0.40178, 0.74029, 0.74725,  0.795, 0.77231, 0.34213, 0.6277, 0.41189, 0.26793, 0.66537, 0.037821, 0.34529, 0.90949, 0.79765, 0.12217, 0.1986, 0.99471, 0.22336, 0.37123, 0.59926, 0.7289, 0.5187, 0.35338, 0.61933,  0.172, 0.44865, 0.85288, 0.73149, 0.37993, 0.54089, 0.39407, 0.38405, 0.72353, 0.94577, 0.7261, 0.61172, 0.38838, 0.35479, 0.59607, 0.047265, 0.36257, 0.86512, 0.080724, 0.040416, 0.56328, 0.81023, 0.52341, 0.81238, 0.34611, 0.8341, 0.41535, 0.35944, 0.49652, 0.84622, 0.4791, 0.87301, 0.27823, 0.91468, 0.31215, 0.80152, 0.42458, 0.77905, 0.37589, 0.011469, 0.64646, 0.47782, 0.46193, 0.92965, 0.44081, 0.90764, 0.2706, 0.57544, 0.48159, 0.041731, 0.069684, 0.34666, 0.84168, 0.45178, 0.31982, 0.42226, 0.37554, 0.88332, 0.53978, 0.61067,  0.265, 0.36836, 0.57646, 0.04954, 0.0091128, 0.2602, 0.62696, 0.93505, 0.71606, 0.22917, 0.65283, 0.7371, 0.29731, 0.0033611, 0.38226, 0.2526, 0.13478, 0.61583, 0.3852, 0.32596, 0.54849, 0.98207, 0.3414, 0.78586, 0.58974, 0.089662, 0.45177, 0.23078, 0.021169, 0.052343, 0.88433, 0.98386, 0.73747, 0.20503, 0.88642, 0.77745, 0.46153, 0.95871, 0.43171, 0.26391, 0.42254, 0.080808, 0.61507, 0.68451, 0.84601, 0.80108, 0.47726, 0.48502, 0.47429, 0.011861, 0.95693, 0.019282, 0.74674, 0.58394, 0.23678, 0.90939, 0.28507, 0.25117, 0.45228, 0.20337, 0.38105, 0.99474, 0.42394, 0.10184, 0.87634, 0.41018, 0.54138, 0.27958, 0.66672, 0.26192, 0.88148, 0.23947, 0.84553, 0.77362, 0.60184, 0.85193, 0.73917, 0.36251, 0.86147, 0.81863, 0.78695, 0.10812, 0.35524, 0.68901, 0.49213, 0.23217, 0.77327, 0.87579, 0.26926, 0.46502, 0.97958, 0.10823, 0.62711, 0.95349, 0.30978, 0.07664, 0.43443, 0.89182, 0.76677, 0.61716, 0.49199, 0.042709, 0.69752, 0.16584, 0.29179, 0.61776, 0.24074, 0.48252, 0.22296, 0.12425, 0.27649, 0.13104, 0.72093, 0.79574, 0.52722, 0.60484, 0.75302, 0.10645, 0.36104, 0.87166, 0.72019, 0.031054, 0.71297, 0.29379, 0.28318, 0.21676, 0.74887, 0.86929, 0.8765, 0.082121, 0.077764, 0.88457, 0.74675, 0.59331, 0.51625, 0.84801,   0.15, 0.029778, 0.13086, 0.47291, 0.45184, 0.034867, 0.85657, 0.71552, 0.67055,  0.257, 0.69702, 0.11856, 0.38103, 0.64477, 0.95058, 0.0088321, 0.058568, 0.74531, 0.079263, 0.27942, 0.78681, 0.40325, 0.3422, 0.91945, 0.44836, 0.2319, 0.48142, 0.41068, 0.45025, 0.6621, 0.97624,  0.114, 0.27016, 0.24669, 0.93501,  0.418, 0.33921, 0.94839, 0.50029, 0.47498, 0.46153, 0.93453, 0.73953, 0.66317, 0.63954, 0.3691, 0.2484, 0.25446, 0.99065, 0.86459, 0.36687, 0.11083, 0.74496, 0.30507, 0.49552, 0.99004, 0.82171, 0.96133, 0.23343, 0.15244, 0.081704, 0.98445, 0.1786, 0.19594, 0.2478, 0.018206, 0.90721, 0.89062, 0.48804, 0.70297, 0.58854, 0.9484, 0.60381, 0.92796, 0.9362, 0.79039, 0.54629, 0.17698, 0.61578, 0.54712, 0.54319, 0.80358, 0.63575, 0.95455, 0.41682, 0.0089834, 0.29845, 0.88679, 0.31299, 0.53449, 0.6245, 0.19656, 0.53816, 0.55056, 0.27177, 0.3072, 0.1889, 0.47422, 0.26588, 0.28347, 0.67729, 0.57338, 0.23341, 0.14875, 0.50949, 0.85249, 0.18767, 0.6847, 0.58474, 0.48013, 0.40954, 0.32026, 0.032783, 0.016003, 0.41236, 0.76756, 0.86809, 0.80633, 0.48211, 0.69331, 0.13379, 0.076666, 0.49717, 0.48743, 0.42808, 0.78089, 0.85464, 0.78281, 0.34033, 0.93633, 0.8616, 0.67356, 0.57441, 0.61265, 0.32377, 0.4952, 0.93833, 0.38883, 0.16313, 0.85279, 0.48146, 0.94872, 0.94459,  0.858, 0.72641, 0.26556, 0.86325, 0.69368, 0.44545, 0.77049, 0.46156, 0.045407, 0.55458, 0.78178, 0.56724, 0.15774, 0.42943, 0.15611, 0.25386, 0.62167, 0.98402, 0.14419, 0.7903, 0.33593, 0.14836, 0.9023, 0.61903, 0.83173, 0.57788, 0.42506, 0.33462, 0.84219, 0.76251, 0.13236, 0.86969, 0.0048131, 0.95656, 0.55942, 0.5412, 0.13184, 0.41412, 0.3562, 0.51093, 0.8148, 0.71726, 0.57786, 0.47986, 0.59751, 0.027196, 0.069291, 0.94445, 0.052346, 0.85842, 0.11829, 0.57442, 0.87155, 0.56724, 0.34825, 0.71068, 0.80698, 0.66807, 0.099578, 0.36859, 0.43496, 0.66035, 0.98583, 0.57232, 0.31478, 0.47237, 0.24133, 0.091209, 0.091424, 0.72093, 0.78054, 0.61388, 0.060385, 0.8441, 0.51711, 0.75746, 0.30763, 0.30353, 0.29665, 0.67904, 0.34489, 0.83628, 0.52204, 0.11352, 0.33087, 0.63181, 0.54567, 0.37148, 0.45178, 0.73882, 0.67247, 0.86439, 0.21263, 0.14024, 0.37164, 0.49301, 0.44601, 0.35804, 0.69035, 0.99344, 0.49457, 0.76816, 0.57123, 0.5332, 0.080506, 0.40398, 0.5852, 0.81737, 0.93071, 0.31496, 0.60523, 0.66168, 0.78444, 0.65153, 0.18143, 0.82046, 0.13169, 0.71883, 0.49423, 0.97621, 0.56909, 0.26106, 0.36913, 0.83713, 0.51946, 0.060447, 0.65492, 0.98329, 0.44789, 0.10617, 0.64156, 0.79572, 0.23264, 0.10017, 0.10447, 0.98419, 0.88286, 0.20097, 0.51976, 0.53292, 0.21551, 0.50414, 0.86379, 0.25202, 0.24725, 0.47138, 0.67348, 0.16664, 0.12138, 0.027113, 0.0087272, 0.69437, 0.13724, 0.0041889, 0.86587, 0.80275, 0.49652, 0.43729, 0.50747, 0.7275, 0.077364, 0.35312, 0.8676, 0.31786, 0.60239, 0.16425, 0.55014, 0.014282, 0.79849, 0.45156, 0.23581, 0.036203, 0.35133, 0.85382, 0.22623, 0.83925, 0.27898, 0.39944, 0.77708, 0.42083, 0.1836, 0.98181, 0.81212, 0.84843, 0.9792, 0.92572, 0.12015, 0.57899, 0.18599, 0.34225, 0.66836, 0.88323, 0.81575, 0.2763, 0.80861, 0.19012, 0.046019, 0.13159, 0.88436, 0.013379, 0.56009, 0.64562, 0.090548, 0.13338, 0.79472, 0.4106, 0.70922, 0.31368, 0.99531,  0.286, 0.4864, 0.84501, 0.14297, 0.85044, 0.087945, 0.63203, 0.0033271, 0.29706, 0.39858, 0.087236, 0.4188, 0.19905, 0.27558, 0.53475, 0.26307, 0.82465, 0.36776, 0.3842, 0.47866, 0.39533, 0.70033, 0.55592, 0.046771, 0.01709, 0.55956, 0.82384, 0.89249, 0.32926, 0.8006, 0.62483, 0.70292, 0.8496, 0.9929, 0.78571, 0.2991, 0.20383, 0.78158, 0.9995, 0.17498, 0.23825, 0.50039, 0.58141, 0.028716, 0.67669, 0.75636, 0.41372, 0.21624, 0.93377, 0.47684, 0.99881, 0.63085, 0.53547, 0.46893, 0.62934, 0.98622, 0.94358, 0.59547, 0.88068, 0.52274, 0.0093893, 0.60258, 0.20742, 0.37381, 0.35441, 0.86059, 0.24293, 0.070364, 0.42327, 0.030893, 0.25393, 0.96727, 0.036675, 0.16735, 0.35305, 0.72504, 0.88587, 0.30674, 0.43439, 0.6346, 0.24072, 0.39528, 0.38484, 0.53075, 0.29698, 0.27938, 0.97356, 0.54834, 0.98782, 0.51166, 0.38247, 0.83736, 0.58786, 0.042733, 0.55822, 0.61059, 0.5073, 0.11418, 0.011553, 0.024302, 0.82435, 0.25723, 0.62554, 0.014256, 0.90172, 0.11802, 0.91217, 0.64018, 0.83175, 0.0013462, 0.77886, 0.87051, 0.92752, 0.39118, 0.80108, 0.48045, 0.62346, 0.71585, 0.43663, 0.38114, 0.1374, 0.92601, 0.43974, 0.45403, 0.36853, 0.93313, 0.72935,  0.818, 0.79041, 0.43663, 0.7236, 0.37881, 0.69534, 0.61196, 0.76519, 0.70527, 0.66108, 0.16989, 0.85512, 0.68527, 0.97277, 0.14062, 0.93602, 0.98275, 0.54631, 0.44828, 0.36962, 0.65309, 0.38018, 0.15311, 0.30388, 0.22885, 0.94554, 0.16355, 0.25498, 0.12503, 0.80177, 0.59655, 0.47808, 0.84997, 0.97341, 0.69278, 0.30527, 0.16376, 0.98378, 0.77041, 0.91848, 0.028544, 0.94727, 0.80405, 0.5332, 0.93194, 0.68566, 0.41648, 0.95921, 0.61731, 0.5165, 0.70978, 0.074281, 0.30046, 0.10091, 0.72556, 0.026783, 0.96619, 0.89372, 0.7572, 0.53343, 0.67793, 0.68966, 0.86786, 0.17727, 0.5315, 0.61781, 0.22277, 0.56857, 0.079914, 0.27286, 0.87005, 0.51915, 0.26497, 0.069432, 0.37777, 0.40889, 0.47748, 0.36538, 0.4911, 0.33808, 0.9007, 0.79867, 0.2362, 0.48291, 0.82338, 0.94245, 0.22477, 0.46194, 0.064313, 0.43288, 0.89112, 0.85213, 0.76678, 0.98229, 0.66669, 0.39441, 0.15712, 0.51949, 0.32166, 0.42372, 0.021433, 0.61333, 0.2184, 0.99012, 0.011255, 0.32594, 0.67395, 0.56195, 0.030905, 0.021238, 0.56555, 0.78503, 0.007517, 0.5589, 0.8959, 0.68004, 0.73794, 0.068363, 0.73318, 0.49433, 0.46636, 0.84745, 0.65526, 0.041288, 0.75144, 0.33117, 0.66464, 0.22597, 0.41105, 0.90232, 0.59097, 0.83201, 0.05451,   0.14, 0.69044, 0.88649, 0.18777, 0.51235, 0.11421, 0.43167, 0.78905, 0.89059, 0.31855, 0.40354, 0.93768, 0.47533, 0.92847, 0.82116, 0.54175, 0.17346, 0.21844, 0.59215, 0.43089, 0.89502, 0.39383, 0.96486, 0.84147, 0.017664, 0.27034, 0.85247, 0.66997, 0.59719, 0.11368, 0.53687, 0.13327, 0.92646, 0.8541, 0.80244, 0.93828, 0.53964, 0.51758, 0.53779, 0.99118, 0.18395, 0.85198, 0.68478, 0.73416, 0.25118, 0.9673, 0.99396, 0.91903, 0.41581, 0.50286, 0.42813, 0.89911, 0.97579, 0.64792, 0.36653, 0.64584, 0.07309, 0.31905, 0.78883, 0.36354, 0.4218, 0.077137, 0.17717, 0.34347, 0.11916, 0.32224, 0.037887, 0.27643, 0.6034, 0.29318, 0.18302, 0.48704, 0.70831, 0.25164, 0.51577, 0.67243, 0.47056, 0.16626, 0.23384, 0.43962, 0.021776, 0.56146, 0.38008, 0.56492, 0.015564, 0.06283, 0.68107, 0.35501, 0.38697, 0.11611, 0.88914, 0.2716, 0.54143, 0.30762, 0.13959, 0.10339, 0.7453, 0.049281, 0.44501, 0.18561, 0.98981, 0.71017,  0.664, 0.80214, 0.84928, 0.37343, 0.26421, 0.4138, 0.51458, 0.76933, 0.67387, 0.022145, 0.72861, 0.13357, 0.42966, 0.87816, 0.015258, 0.85864, 0.34406, 0.84212, 0.98258, 0.020407, 0.38611, 0.87732, 0.47521, 0.086127, 0.43331, 0.62529, 0.30438, 0.73871, 0.83738, 0.59808, 0.11301, 0.17005, 0.059031,  0.914, 0.75977, 0.9358, 0.99454, 0.34524, 0.44608, 0.4494, 0.26978, 0.38721, 0.88852, 0.54227, 0.22199, 0.28053, 0.95298, 0.63825, 0.26445, 0.13484, 0.6211, 0.13323, 0.86228, 0.13067, 0.051568, 0.64674, 0.92469, 0.65237, 0.57067, 0.64937, 0.24392, 0.70795, 0.22079, 0.26579, 0.81614, 0.75951, 0.81215, 0.92073, 0.42071, 0.70547, 0.88011, 0.18475, 0.15634, 0.65769, 0.21081, 0.57372, 0.12461, 0.097144, 0.058914, 0.8987, 0.71479, 0.87201, 0.94059, 0.067321, 0.5058, 0.55927, 0.53208, 0.059814, 0.55504, 0.35701, 0.20544, 0.67737, 0.59725, 0.18834, 0.59547, 0.13014, 0.38793, 0.98114, 0.4601, 0.073952, 0.78567, 0.19729, 0.93623, 0.39948, 0.69625, 0.016953, 0.8305, 0.0066127, 0.45658, 0.66696, 0.024524, 0.41951, 0.1359, 0.64276, 0.41577, 0.87122, 0.021063, 0.79077, 0.65726, 0.18391, 0.82658, 0.44734, 0.50338, 0.38037, 0.27225, 0.1136, 0.36731, 0.37464, 0.3827, 0.24819, 0.19199, 0.72599, 0.60752, 0.86072, 0.39486, 0.94275, 0.3568, 0.30234, 0.5939, 0.50152, 0.85571, 0.011029, 0.16097, 0.22061, 0.53248, 0.78094, 0.79488, 0.47386, 0.56726, 0.30323, 0.87169, 0.83451, 0.75218, 0.66566, 0.24039, 0.89528, 0.84706, 0.24811, 0.9263, 0.32533, 0.072541, 0.7341, 0.13974, 0.75727, 0.4908, 0.99261, 0.40886, 0.038665, 0.36065, 0.47224, 0.74662, 0.067208, 0.38032, 0.62886, 0.23728, 0.95347, 0.8286, 0.080229, 0.19471, 0.58854, 0.77106, 0.44585, 0.048636, 0.3008, 0.18194, 0.69178, 0.23575, 0.35564, 0.70457, 0.1742, 0.26184, 0.64512, 0.58017, 0.67736, 0.76367, 0.95905, 0.51467, 0.7035, 0.70602, 0.063945, 0.35533, 0.5356, 0.34221, 0.6396, 0.66897, 0.19996, 0.12457, 0.11061, 0.79249, 0.79926, 0.69631, 0.28131, 0.52814, 0.12045, 0.50143, 0.85742, 0.88457, 0.67677, 0.71071, 0.71497, 0.66677, 0.29249, 0.51608, 0.89514, 0.35365, 0.2267, 0.26553, 0.92735, 0.45582, 0.99655, 0.42709, 0.90278, 0.038453, 0.86778, 0.86201, 0.59375, 0.12819, 0.44621, 0.18105, 0.69163, 0.47768, 0.49838, 0.59465, 0.71858, 0.64942, 0.6148, 0.66813, 0.068157,  0.175, 0.42908, 0.91081, 0.82787, 0.32511, 0.25177, 0.42388, 0.62978, 0.78054, 0.79892, 0.27016, 0.98394, 0.30355, 0.27522, 0.13507,  0.445, 0.30495, 0.89753, 0.42906, 0.37082, 0.62311, 0.48895, 0.14508, 0.51583, 0.35578, 0.7436, 0.068865, 0.79636, 0.28842, 0.099409, 0.93189, 0.30643, 0.80593, 0.80591, 0.16675, 0.56076, 0.18776, 0.58232, 0.6716, 0.21749, 0.99035, 0.22332, 0.45667, 0.17192, 0.90536, 0.6766, 0.87068, 0.61883, 0.8908, 0.14564, 0.58736, 0.81814, 0.44109, 0.35235, 0.56659, 0.040615, 0.30701, 0.26135, 0.45594,  0.123, 0.58786, 0.62619, 0.18375, 0.72581, 0.65214, 0.52119, 0.17285, 0.1319, 0.35696, 0.19628, 0.075619, 0.68594, 0.18363, 0.54304, 0.34151, 0.084492, 0.43918, 0.86296, 0.42266, 0.36234, 0.56585, 0.54445, 0.015142, 0.71686, 0.35221, 0.43268, 0.71302, 0.9436, 0.88367, 0.39586, 0.38083, 0.66806, 0.27719, 0.88112, 0.51406, 0.14311, 0.33437, 0.2785, 0.99428, 0.1664, 0.040893, 0.95815, 0.29205, 0.81612, 0.19622, 0.84951, 0.38314, 0.86835, 0.062369, 0.83839, 0.082434, 0.43913, 0.45112, 0.50741, 0.75179, 0.33899, 0.95764, 0.37088, 0.09911, 0.18707, 0.60259, 0.027285, 0.33381, 0.16615, 0.2713, 0.94721, 0.45596, 0.78849, 0.85893, 0.40383, 0.32596, 0.020177, 0.46918, 0.065337, 0.87908, 0.54219, 0.033149, 0.8106, 0.25872, 0.80714, 0.58635, 0.38477, 0.37916, 0.024732, 0.014962, 0.99535,  0.261, 0.53551, 0.98827, 0.67131, 0.49021, 0.68924, 0.24312, 0.34011, 0.39012, 0.95149, 0.75213, 0.055073, 0.63957, 0.46596, 0.73408, 0.79535, 0.58478, 0.45169, 0.18653, 0.67487, 0.38122, 0.87092, 0.32318, 0.71151, 0.13585, 0.64512, 0.40187, 0.15647, 0.57652, 0.63875, 0.30822, 0.60925, 0.75972, 0.75641, 0.075588, 0.66276, 0.92734, 0.51633, 0.37622, 0.79982, 0.58481, 0.28578, 0.46302, 0.73649, 0.94001, 0.46165, 0.22077, 0.56041, 0.72849, 0.27719, 0.88656, 0.055731, 0.93309, 0.88046, 0.43392, 0.16784, 0.74968, 0.30008, 0.29936, 0.17529, 0.12113, 0.37417, 0.70711, 0.36683, 0.85416, 0.93976, 0.51971, 0.18343, 0.070225, 0.9548, 0.52212, 0.66269, 0.12548, 0.19668, 0.09037, 0.41576, 0.49844, 0.5854, 0.99738, 0.10219, 0.12591, 0.72386, 0.45646, 0.94835, 0.58576, 0.1447, 0.20334, 0.22678, 0.22369, 0.75667, 0.68608, 0.39017, 0.6047, 0.54925, 0.44511, 0.91374, 0.59666, 0.95831, 0.62752, 0.9137, 0.43654, 0.90952, 0.4775, 0.2845, 0.19131, 0.69392, 0.6312, 0.58692, 0.99936, 0.94412, 0.79187, 0.5436, 0.4024, 0.21786, 0.27093, 0.66896, 0.093311, 0.67675, 0.3427, 0.15984, 0.51927, 0.64891, 0.72343, 0.8991, 0.080934, 0.092427, 0.92393, 0.58249, 0.34121, 0.98461, 0.70585, 0.70783, 0.74709, 0.7979, 0.70021, 0.42386, 0.65512, 0.17128, 0.85662, 0.52822, 0.0011302, 0.60498, 0.38513, 0.94529, 0.82166, 0.90346, 0.80393, 0.81065, 0.58252, 0.56689, 0.35284, 0.99554, 0.30222, 0.64444, 0.069991, 0.87378, 0.60719, 0.56203, 0.81939, 0.47168, 0.99524, 0.7728, 0.71687, 0.24239, 0.050274, 0.90666, 0.38055, 0.13584, 0.15269, 0.64844, 0.10477, 0.68103, 0.41415, 0.56344, 0.27342, 0.87862, 0.1565, 0.42201, 0.94443, 0.61675, 0.44494, 0.23345, 0.43998, 0.0002502, 0.83553, 0.91598,  0.303, 0.23452, 0.92793, 0.87739, 0.22222, 0.52797, 0.028463, 0.73261, 0.49337, 0.39469, 0.27586, 0.49053, 0.19898, 0.66673, 0.11455, 0.098127, 0.39722, 0.059995, 0.13963, 0.95296, 0.5799, 0.75944, 0.1758, 0.59746, 0.62062, 0.65028, 0.84542, 0.064226, 0.51461, 0.54863, 0.56229, 0.074063, 0.38655, 0.0099331, 0.30299, 0.5346, 0.58332, 0.88915, 0.46987, 0.65525, 0.3645, 0.40129, 0.1782, 0.1863, 0.027236, 0.66309, 0.00043755, 0.45637, 0.27077, 0.91935, 0.33829, 0.65402, 0.31256, 0.22893, 0.35408, 0.94024, 0.087478, 0.067773, 0.69916, 0.036899, 0.2789, 0.47341, 0.66883, 0.72598, 0.27198, 0.79531, 0.8482, 0.52327, 0.73207, 0.052493, 0.45898, 0.64119, 0.1001, 0.021399, 0.03811, 0.1258, 0.62506, 0.095375, 0.45162, 0.37406, 0.76446, 0.76063, 0.037273, 0.20842, 0.077794, 0.22143, 0.33099, 0.016884, 0.50644, 0.89342, 0.17755, 0.12224, 0.57147, 0.66369, 0.61632, 0.14398, 0.52108, 0.084128, 0.93922, 0.22978,   0.54, 0.90521, 0.40713, 0.99196, 0.1266, 0.68884, 0.05807, 0.72977,  0.252, 0.57916, 0.45018, 0.50082, 0.81879, 0.82169, 0.40267, 0.027889, 0.63571, 0.73974, 0.027876, 0.51074, 0.82741, 0.29419, 0.055305, 0.074295, 0.58866, 0.6622, 0.91131, 0.70633, 0.63009, 0.35264, 0.63711, 0.65445, 0.82794, 0.44134, 0.88773, 0.96171, 0.35874, 0.50153, 0.78392, 0.70614, 0.87446, 0.98802, 0.67769, 0.65969, 0.081749, 0.36905, 0.52545, 0.67375, 0.42082, 0.69974, 0.22128, 0.61225, 0.2792, 0.065171, 0.62605, 0.18272, 0.45926, 0.7405, 0.37818, 0.95981, 0.71683, 0.23879, 0.74467, 0.32398, 0.16263, 0.26164, 0.67333, 0.028395, 0.21282, 0.73489, 0.95923, 0.78151, 0.50673, 0.5234, 0.65406, 0.60882, 0.28937, 0.9196, 0.077817, 0.87149, 0.90332, 0.44396, 0.35026, 0.63371, 0.16058, 0.84153, 0.84918, 0.6753, 0.96189, 0.42752, 0.22295, 0.11426, 0.35068, 0.8867, 0.32923, 0.094713, 0.11793, 0.80008, 0.44584, 0.083949, 0.85021, 0.56736, 0.25138, 0.89112,  0.289, 0.16925, 0.99557, 0.51745, 0.63762, 0.49296, 0.67789, 0.18392, 0.11597, 0.11321, 0.62897, 0.58627, 0.65181, 0.0069028, 0.68116, 0.2623, 0.62229, 0.60815, 0.32106, 0.9535, 0.0033391, 0.69264, 0.41569, 0.88376, 0.64441, 0.68166, 0.4997, 0.46982, 0.3248,  0.249, 0.81814, 0.70631, 0.43319, 0.076408, 0.00068, 0.91947, 0.39872, 0.6266, 0.6121, 0.98164, 0.48788, 0.54405, 0.79536, 0.54083, 0.25814, 0.80174, 0.67646, 0.7971, 0.65097, 0.55208, 0.7492, 0.93793, 0.82342, 0.65373, 0.74167, 0.21176, 0.84294, 0.50725, 0.81619, 0.72259, 0.96302, 0.73816, 0.48653, 0.037678, 0.32512, 0.71026, 0.75409, 0.27802, 0.33434, 0.92792, 0.53179, 0.41981, 0.2589, 0.50573, 0.47007, 0.45261, 0.84458, 0.7362, 0.38275, 0.048925, 0.57601, 0.33533, 0.11113, 0.48054, 0.083332, 0.16194, 0.18598, 0.099373, 0.39646, 0.92195, 0.12915, 0.28585, 0.31602, 0.038331, 0.35817, 0.44845, 0.29622, 0.59025, 0.23291, 0.12436, 0.16664, 0.029138, 0.86814, 0.37614, 0.23792, 0.49271, 0.62691, 0.038292, 0.13138, 0.19958, 0.88027, 0.65829, 0.97035, 0.82656, 0.74172, 0.052846, 0.81132, 0.43248, 0.89793, 0.29447, 0.95609, 0.98262 };
const int WLENGTH = 3000;
void nonRandomWeightsTConv(Layer* layer)
{
    auto& weights = layer->getParams();

    for (int i = 0; i < weights.size(); ++i)
    {
        auto stdv = 1. / std::sqrt(6. / 28.);
        auto stdv2 = stdv * 2;
        auto d = WEIGHTS[i % WLENGTH] * stdv2 - stdv;
        weights[i] = d;
    }
}
void nonRandomWeightsFC(Layer* layer)
{

    int inputLength;
    if (layer->getType() == "FullyConnectedLayerBlas")
    {
        FullyConnectedLayerBlas* fc = (FullyConnectedLayerBlas*)layer;
        inputLength = fc->getInputLength();
    }
    else
    {
        FullyConnectedLayer* fc = (FullyConnectedLayer*)layer;
        inputLength = fc->getInputLength();
    }
    auto& weights = layer->getParams();
    for (int i = 0; i < weights.size(); ++i)
    {
        double stdv = 1. / std::sqrt(inputLength);
        double stdv2 = stdv * 2;
        double d = WEIGHTS[i % WLENGTH] * stdv2 - stdv;
        weights[i] = d;
    }
}


void WeightHacks::shuffle(std::vector<sgdtk::FeatureVector*>& list)
{
    int sz = list.size();

    for (int i = 0; i < sz; i += 4)
    {
        int j = sz-i-1;
        std::swap(list[j], list[i]);
    }
}

void WeightHacks::hack(NeuralNetModel* nnModel)
{
    auto layers = nnModel->getLayers();

    for (Layer* layer : layers)
    {
        String type = layer->getType();

        if (type =="SpatialConvolutionalLayer" || type=="TemporalConvolutionalLayer" || type == "TemporalConvolutionalLayerFFT")
        {
            nonRandomWeightsTConv(layer);
        }
        else if (type == "FullyConnectedLayer" || type=="FullyConnectedLayerBlas")
        {
            nonRandomWeightsFC(layer);
        }
    }
}
