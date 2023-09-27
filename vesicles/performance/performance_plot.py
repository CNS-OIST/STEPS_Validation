# MPI performance of a realistic synaptic vesicle cycle model in STEPS 5.0

from matplotlib import pyplot as plt

cores=[2,4,8,16,32,64,128,256,384,512]
rt_2000 = [385.680873, 108.572092, 53.558429, 31.449047, 24.655371, 15.559180, 14.856715, 9.431283, 9.244792, 7.279367]

plt.plot(cores, rt_2000, linewidth=5)

ax0=plt.gca()
ax0.set_xscale('log')
ax0.set_yscale('log')
plt.xlabel("Number of cores", fontsize=12)
plt.ylabel("Mean wall-clock runtime (s)", fontsize=12)
plt.ylim(6,450)
ax0.set_yticks([], minor=True)

ax0.set_yticks((10,20, 50,100,200, 400))
ax0.set_yticklabels(('10','20','50','100', '200', '400'))

ax0.set_xticks([], minor=True)
ax0.set_xticks((2,4,8,16,32,64,128,256,512))
ax0.set_xticklabels(('2','4','8','16','32','64','128','256','512'))

plt.savefig("performance.pdf")
