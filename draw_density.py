import matplotlib.pyplot as plt


data_density = [0.3501517916429101, 0.3495802895431938, 0.34866973903558823, 0.34740879130694596, 0.34576805633899577, 0.3437444388429395, 0.34133469561455576, 0.33853712530984376, 0.3353418503082602, 0.3317381756616173, 0.327713239921191, 0.3232510900090152, 0.31835303247789765, 0.31300343494929056, 0.3071588587960094, 0.30079653735920414, 0.29388737146188726, 0.2864080874605971, 0.2783543935201476, 0.26970069374092204, 0.2604446149143416, 0.25061069539757336, 0.240203470288555, 0.2292681798461118, 0.21785422915166996, 0.2060288785666547, 0.1938687457093719, 0.1814789301946578, 0.16896666501993826, 0.15643923118635283, 0.1440120362462243, 0.13179720404180326, 0.11990056234834248, 0.10842141373327911, 0.09744613414291915, 0.08704582740193752, 0.07727698506078029, 0.06818170987935616, 0.05978695609094881, 0.052103982490872496, 0.04513068477341417, 0.03885287854531781, 0.0332463469554898, 0.028278414406127782, 0.02391002989260616, 0.02009756787535332, 0.01679466525050841, 0.013953796080895304, 0.011527516224267682, 0.009469594234585064, 0.007735888059479121, 0.006284974851984389, 0.005078593471985476, 0.004081897538291218, 0.0032635634975543243, 0.002595770905138947, 0.0020540836263977617, 0.0016172624068521496, 0.0012670296908139378, 0.0009878031085668543, 0.000766416167856317, 0.0005918381539023663, 0.000454902782205267, 0.0003480525050095375, 0.0002651029416474466, 0.00020102998212874454, 0.00015178061201852748, 0.0001141072813438308, 8.542482870877118e-05, 6.368839773165247e-05]
data_xlabel = []
for i in range(1,71):
    x1 = i*0.05
    data_xlabel.append(x1)

plt.plot(data_xlabel,data_density,linewidth=1)

data2 = [[0.05056179775280906, 0.3551948051948051],
[0.12359550561797761, 0.35454545454545444],
[0.19662921348314621, 0.3519480519480519],
[0.2724719101123596, 0.3487012987012986],
[0.351123595505618, 0.3448051948051948],
[0.4241573033707865, 0.3402597402597402],
[0.5, 0.33441558441558433],
[0.5758426966292135, 0.32662337662337654],
[0.6516853932584269, 0.3181818181818181],
[0.7303370786516853, 0.309090909090909],
[0.8061797752808989, 0.29870129870129863],
[0.8820224719101123, 0.28831168831168824],
[0.9578651685393259, 0.27532467532467525],
[1.0365168539325844, 0.2616883116883116],
[1.1151685393258428, 0.24610389610389605],
[1.1938202247191012, 0.22922077922077916],
[1.2724719101123596, 0.21103896103896103],
[1.3483146067415728, 0.19415584415584414],
[1.4325842696629212, 0.17337662337662335],
[1.5140449438202244, 0.15389610389610392],
[1.595505617977528, 0.13376623376623364],
[1.674157303370786, 0.11493506493506489],
[1.7584269662921344, 0.09740259740259727],
[1.8426966292134832, 0.08116883116883111],
[1.929775280898876, 0.06558441558441563],
[2.0140449438202244, 0.051948051948051854],
[2.101123595505618, 0.040909090909090895],
[2.185393258426966, 0.03116883116883118],
[2.280898876404494, 0.02207792207792214],
[2.3707865168539324, 0.016233766233766267],
[2.46629213483146, 0.011688311688311637],
[2.5617977528089884, 0.007792207792207684],
[2.660112359550561, 0.004545454545454519],
[2.7584269662921344, 0.0032467532467532756],
[2.859550561797753, 0.00194805194805181],
[2.96629213483146, 0.00194805194805181]]

x_values = []
y_values = []
for i in range(len(data2)):
    x_values.append(data2[i][0])
    y_values.append(data2[i][1])

plt.scatter(x_values,y_values,c='red',s=5)

plt.show()
