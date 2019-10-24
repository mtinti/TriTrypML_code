def compute(indata):
    import pandas as pd
    import sys
    sys.path.append("profet/")
    from profet.FeatureGen import Get_Protein_Feat
    conv_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    ref_aas = set(conv_dict.values())
    
    #test if the proteins contains only
    #canonical AA
    def check_aminoacids(in_seq):
        temp_aas = set(list(in_seq))
        res = temp_aas-ref_aas
        if len(res) >0:
            return False
        else:
            return True 
    
    #test if the proteins contains *
    def check_pseudo(in_seq):
        if '*' in in_seq:
            return False
        else:
            return True
    
    #test if the starts with a M
    def check_M(in_seq):
        if 'M' == in_seq[0]:
            return True
        else:
            return False
    
    sequence = indata[0]
    temp_id = indata[1]
    #print (check_M(sequence))
    if( check_M(sequence) and 
    check_pseudo(sequence) and 
    check_aminoacids(sequence) and len(sequence)>=30): 
        d = Get_Protein_Feat(sequence)
        s = pd.Series(d)
        s.loc['protein']=temp_id
        return s
    else:
        s = pd.Series()
        s.loc['protein']=temp_id
        return s