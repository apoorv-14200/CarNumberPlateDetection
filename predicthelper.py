import finalpreprocessing as seg


def rectify(s):
    s = list(s)
    for i in range(2):
        if s[i]=='4':
            s[i]='A'
        if s[i]=='2':
            s[i] = 'Z'
        if s[i]=='1':
            s[i] = 'I'
        if s[i]=='8':
            s[i] = 'B'
        if s[i]=='6':
            s[i] = 'G'
        if s[i]=='0':
            s[i] = 'O'
        if s[i]=='5':
            s[i] = 'S'
    for i in range(2,min(4,len(s))):
        if s[i]=='A':
            s[i]='4'
        if s[i]=='Z':
            s[i] = '2'
        if s[i]=='I':
            s[i] = '1'
        if s[i]=='B':
            s[i] = '8'
        if s[i]=='G':
            s[i] = '6'
        if s[i]=='O':
            s[i] = '0'
        if s[i]=='S':
            s[i] = '5'   
            
    for i in range(6,min(len(s),10)):
        if s[i]=='A':
            s[i]='4'
        if s[i]=='Z':
            s[i] = '2'
        if s[i]=='I':
            s[i] = '1'
        if s[i]=='B':
            s[i] = '8'
        if s[i]=='G':
            s[i] = '6'
        if s[i]=='O':
            s[i] = '0'
        if s[i]=='S':
            s[i] = '5'      
    s = ''.join(s)   
    return s


def predict(im,reader,ALLOWED_LIST):
    im=seg.getprocessed(im)
    characters = reader.readtext(im, allowlist=ALLOWED_LIST)
    characters.sort(key=lambda x:(x[0][0][0]+x[0][2][0])//2)
    p=""
    for i in range(len(characters)):
        p+=characters[i][1]

    return p

