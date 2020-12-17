infn = r"D:\Daniel\Dropbox (MIT)\Projects\RFQ Direct Injection\Emittance Scanners\beam15uA_3_mod2.csv"
outfn = r"D:\Daniel\Dropbox (MIT)\Projects\RFQ Direct Injection\Emittance Scanners\out2.txt"

with open(infn, "r") as infile:
    lines = infile.readlines()

with open(outfn, "w") as of:
    for line in lines:
        of.write("{}\n".format(line.strip().replace(",", "\t")))
