from multiprocessing import Pool

def f(x):
	return x**2

if __name__=="__main__":
	with Pool() as p:
		print(p.map(f, [1, 2, 4, 6])[0])
