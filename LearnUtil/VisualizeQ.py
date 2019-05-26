import pickle
import matplotlib.pyplot as plt
import numpy as np
def load_loss(path, epoch):
    try:
        with open(path + "_QValues_" + str(epoch), 'rb') as handle:
            return pickle.load(handle)
    except:
        return None

def plot_loss(path, epoch=None, start_ep = 0, epochjump=450):
    b = {}
    # if epoch is None:
    #     epoch = self.var_file(read=True)
    temp=start_ep
    while temp < epoch:
        temp += epochjump
        q = load_loss(path, temp)
        if q is None:
            break
        b.update(q)

    #b=b[start_ep:]
    #x = list(range(epoch + 1 - len(b), epoch + 1))
    for ac in range(4):
        x=[]
        y=[]
        z=[]
        i=start_ep
        for ep in b.values():
            j=0
            for step in ep.values():
                if j%2==0:
                    j+=1
                    continue
                x.append(i)
                y.append(j)

                z.append(step[ac])
                j+=1
            i+=1
        y=list(y)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax=Axes3D(fig)
        ax.plot(xs=x, ys=y, zs=z, zdir='z')

        #plt.plot(list(range(epoch + 1 - len(vals), epoch + 1)), vals, label=path.split('/')[-1][10:])
        plt.title("Q Values")
        #plt.text(x=200+start_ep, y=0.0114, s='Ilość epok', fontdict={'size': 10})
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.legend()
        plt.show()

plot_loss(path='../weights/Pacman/PacmanModel', epoch=25650, start_ep=23850)