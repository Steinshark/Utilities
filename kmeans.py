f = open("10Clusters.csv",'w')
from random import uniform, randint
f.write(F"X,Y\n")
bounds = [-100,100]
clusters = [(uniform(bounds[0],bounds[1]),(uniform(bounds[0],bounds[1]))) for i in range(5)]

print(clusters)
for i in range(1000):
    cl = clusters[randint(0,len(clusters)-1)]
    x = uniform(cl[0]-(bounds[0]/50),cl[1]-(bounds[1]/55))
    y = uniform(cl[0]-(bounds[0]/50),cl[1]-(bounds[1]/55))
    f.write(f"{x},{y}\n")
f.close()



df = pandas.read_csv("10Clusters.csv",sep=",")
df.apply(sum,axis=1)
#sum_vect = numpy.vectorize(sum)
#sum_vect(df,1)
