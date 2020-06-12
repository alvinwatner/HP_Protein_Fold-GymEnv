char = []
string = "ppphhpphhppppphhhhhhhpphhpppphhpphpp"
for i in range(len(string)):
	if string[i] == 'p' or string[i] == 'P':
		char.append('P')
	else:
		char.append('H')

print(char)