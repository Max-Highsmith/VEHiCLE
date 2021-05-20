#this is an autograd differentiable implementation of the insulation score
import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb
import sys
class computeInsulation(torch.nn.Module):
    def __init__(self, window_radius=10, deriv_size=10):
        super(computeInsulation, self).__init__()
        self.window_radius = window_radius
        self.deriv_size    =  deriv_size
        self.di_pool       = torch.nn.AvgPool2d(kernel_size=(2*window_radius+1), stride=1) #51
        self.top_pool      = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool   = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)

    def forward(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,self.deriv_size:])
        bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
        dv     = (top-bottom)
        left   = torch.cat([torch.zeros(dv.shape[0], dv.shape[1],2), dv], dim=2)
        right  = torch.cat([dv, torch.zeros(dv.shape[0], dv.shape[1],2)], dim=2)
        band   = ((left<0) == torch.ones_like(left)) * ((right>0) == torch.ones_like(right))
        band   = band[:,:,2:-2]
        boundaries = []
        for i in range(0, band.shape[0]):
            cur_bound = torch.where(band[i,0])[0]+self.window_radius+self.deriv_size
            boundaries.append(cur_bound)
        return iv, dv, boundaries

def writeToBedPe(chros, starts, ends, fn):
    fn_file = open(fn,'w')
    fn_file.write("chromosome1\tx1\tx2\tchromosome2\ty1\ty2\n")
    for c, (chro,start,end) in enumerate(zip(chros, starts, ends)):
        fn_file.write("".join(["chr",
            str(chro),
            "\t",
            str(start),
            "\t",
            str(end),
            "\t",
            "chr",
            str(chro),
            "\t",
            str(start),
            "\t",
            str(end),
            "\n"]))

if __name__ == "__main__":
    print("np file:",   sys.argv[1])
    print("chro:",      sys.argv[2])
    print("coords",     sys.argv[3])
    print("res",        sys.argv[4])
    print("fn",         sys.argv[5])

    hic_numpy          = np.load(sys.argv[1]).astype(float)
    chro               = sys.argv[2]
    coords             = np.load(sys.argv[3])
    res                = int(sys.argv[4])
    fn                 = sys.argv[5]

    hic_torch          = torch.from_numpy(hic_numpy)
    hic_torch          = torch.unsqueeze(torch.unsqueeze(hic_torch, dim=0), dim=0)
    hic_torch          = hic_torch.float()
    insulationComputer = computeInsulation()
    Tad_Results        = insulationComputer.forward(hic_torch)

    bin_pos = np.array(list(map(lambda x: coords[x], np.array(Tad_Results[2][0]))))
    starts  = bin_pos*res
    ends    = starts+res
    chros   = np.repeat(chro,len(starts)) 
    writeToBedPe(chros=chros,
            starts=starts,
            ends=ends,
            fn=fn)
