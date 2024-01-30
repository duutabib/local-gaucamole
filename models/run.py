from multiprocessing import Pool
import defs

if __name__=="__main__":
	with Pool() as p:
		print(p.map(defs.f, [1, ]))

