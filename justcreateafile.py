h_, w, h = 34, 67, 67

'''
with open('h_wh.txt','w') as f:
    f.write(str(h_))
    f.write('\n')
    f.write(str(w))
    f.write('\n')
    f.write(str(h))
    f.write('\n')
'''
with open('h_wh.txt','r') as f:
    a = f.readlines()
print(eval(a[0]+a[1]+a[2]))

#print(eval(a[2].strip()))
