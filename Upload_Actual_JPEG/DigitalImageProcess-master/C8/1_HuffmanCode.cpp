#if 0
#include<iostream>
#include<conio.h> 
#include<string>
#include<vector>
#include<fstream>
#include<queue>
#include<ctime> 
using namespace std;
long node[256];//���ڴ洢ASCII�����ļ��г��ֵ�Ƶ�� 
string m[256];//Huffman������ASCIIƥ������       
long FileLength = 0;//ͳ���ļ����ȴ����ļ���β�ֽ�ƥ������ 
/*
Huffman���ڵ�Ľ�����
weight:�ڵ��Ȩ��  code:�ýڵ�����Դ�룬left��right:���Һ���
*/
struct Tree
{
	long long weight;
	int code;
	Tree* left;
	Tree* right;
	Tree(long long w, int c, Tree* a, Tree* b) :weight(w), code(0), left(NULL), right(NULL) { weight = w; left = a; right = b; code = c; }
};
/*
Huffman�����������ȶ��бȽϽṹ��
���ڹ������ȶ��е�����˳�򣨰���Ƶ����С�������У�
*/
struct cmp
{
	bool operator()(Tree* a, Tree* b)
	{
		return a->weight > b->weight;
	}
};
//ͳ��ASCII����ַ����ļ��г��ֵ�Ƶ�ʺ��� 
void statistics(char* a)
{
	FileLength = 0;
	ifstream INPUT(a, ios::binary);                                              //���ļ� 
	if (INPUT.fail())return;                                                     //����ļ��������򷵻� 
	while (!INPUT.eof())                                                         //ѭ����ȡÿһ�ֽڣ�����temp�в�ͳ��Ƶ�� 
	{
		unsigned char temp;
		INPUT.read((char*)&temp, sizeof(unsigned char));
		int aa = (int)temp;
		if (!INPUT.eof()) { node[(int)temp]++; FileLength++; }                   //���δ�����ļ�β����Ӷ�Ӧ��Ƶ�ʣ�ͬʱ�ļ�����ͳ�Ƽ�һ 
	}
	INPUT.close();                                                              //�ر��ļ� 
}
//Huffman���뺯�� 
void encode(Tree* root, string a)                                                //�������еݹ�������� 
{
	if (root->left == NULL && root->right == NULL) 
	{ 
		m[root->code] = a; 
	}                //���������Ҷ�ӽڵ㣬��ƥ����Ϣ����m�� 
	else {
		if (root->left != NULL) {                                            //�������Ӳ����ַ�����'0�� 
			a += '0';  encode(root->left, a);
		}
		if (root->right != NULL) {                                           //�����Һ��Ӳ����ַ�����'1' 
			a = a.substr(0, a.length() - 1); a += '1';  encode(root->right, a);
		}
	}
}
//����Huffman������ 
void creatTree(priority_queue<Tree*, vector<Tree*>, cmp>& forest)
{
	for (int i = 0; i < 256; i++)
	{
		if (node[i] != 0)                                                    //����0��Huffman����ͳ�������Խڵ����ʽ���뵽������ 
		{
			Tree* temp = new Tree(node[i], i, NULL, NULL);                   //�����½ڵ� 
			forest.push(temp);                                          //��������� 
		}
	}
	while (1)
	{
		if (forest.size() == 1) 
			break;                                      //ʼ��ѭ��ֱ��������ʣ��һ���ڵ�Ϊֹ��������� 
		Tree* ltemp = forest.top();                                        //�����ȶ��еĶ�ǰ��Ԫ��ȡ����pop�����������ڵ���Ϊ�½ڵ�����Һ��� 
		forest.pop();
		Tree* rtemp = forest.top();
		forest.pop();
		Tree* temp = new Tree(ltemp->weight + rtemp->weight, 0, ltemp, rtemp);  //�½ڵ��Ƶ��Ϊ�������ӽڵ��Ƶ��֮�ͣ��½ڵ㲻��Ҫ�����κ�Դ�� 
		forest.push(temp);

	}
}
/*ѹ������
INPUT:��Դ�ļ�  OUTPUT:��ѹ���ļ�
inByte����ʱ�洢�Ķ����ֽ� outByte����ʱ�洢��д���ֽ�
storage���ȴ���д�����Ϣ  f:�ж��Ƿ������ѹ��
��������Ϊ��ͳ��Ƶ�ʡ�����Huffman������Huffman�����б��롢�ض��ļ����ζ�Ӧд���±��뼰���λ
*/
void compress(char* a, char* b, int& f)
{
	ifstream INPUT(a, ios::binary);                                             //��Դ�ļ� 
	if (INPUT.fail()) { f = 0; cout << "�Բ�������ѹ���ļ������ڣ�" << endl; return; }
	ofstream OUTPUT(b, ios::binary);                                            //��ѹ���ļ�Ԥ��д�� 
	if (OUTPUT.fail())return;
	statistics(a);                                                             //ͳ���ַ����ֵ�Ƶ�ʼ��ļ����� 
	if (FileLength == 0 || FileLength == 1) { f = 0; cout << "�����ļ�Ϊ���ļ�������ѹ����" << endl; return; }
	else if (FileLength <= 10) { cout << "������ʾ�������ļ�ΪС�ļ���ѹ��Ч����������룡" << endl; }
	priority_queue<Tree*, vector<Tree*>, cmp> forest;
	creatTree(forest);                                                         //���ȶ��н���Huffman��
	string str = "";
	encode(forest.top(), str);                                                  //���ݽ�����Huffman������Huffman���� 
	OUTPUT.write((char*)&FileLength, sizeof(FileLength));                       //д��ͳ�ƺõ��ļ�����д�� 
	for (int i = 0; i < 256; i++)                                                     //���Ƚ�ͳ��Ƶ����Ϣд���ļ����Ա��ѹʹ�� 
	{
		OUTPUT.write((char*)&node[i], sizeof(node[i]));
	}
	string storage = "";
	unsigned char inByte, outByte;
	while (!INPUT.eof()) {
		while (storage.length() < 8)                                       //ɨ���ļ�����storage���洢��Ϣ������λ����ѭ��ִ��д�� 
		{
			if (INPUT.eof())break;
			INPUT.read((char*)&inByte, sizeof(inByte));
			if (!INPUT.eof())storage += m[(int)inByte];                   //����׼��д���Huffman����                   
		}
		while (storage.length() >= 8)                                      //д����� 
		{
			outByte = '\0';                                            //��ʼ��д���ֽ� 
			for (int i = 0; i < 8; i++)                                    //��λ�����ֽ���Ϣ 
			{
				outByte <<= 1;                                          //�����ֽ� 
				if (storage[i] == '1') outByte |= 1;                       //�����д��1 
			}
			OUTPUT.write((char*)&outByte, sizeof(outByte));           //д���ֽ� 
			storage = storage.substr(8);                                //д��8λ��ص�storage��ǰ8λ 
		}
		if (INPUT.eof() && storage.length() != 0)                            //�����ȡ���ļ���β��δ��ɱ���д�� 
		{
			outByte = '\0';
			for (int i = 0; i < storage.length(); i++)                    //ʣ���λ��д�� 
			{
				outByte <<= 1;
				if (storage[i] == '1')outByte |= 1;
			}
			outByte <<= (8 - storage.length());                         //����8λ���� 
			OUTPUT.write((char*)&outByte, sizeof(outByte));         //д�����һλ�ֽ� 
		}
	}
	INPUT.close();
	OUTPUT.close();

}
/*��ѹ����
INPUT:��ѹ���ļ�  OUTPUT:�򿪽�ѹ�ļ�
inByte����ʱ�洢�Ķ����ֽ�   outByte: ��ʱ�洢��д���ֽ�
p������Huffman����Ҫ��ָ��   f:�ж��Ƿ�����˽�ѹ
��������Ϊ����ȡƵ�ʣ����½���������Դ�룬д���ļ�
*/
void uncompress(char* a, char* b, int& f)
{
	long long WriteLength = 0;                                                    //ͳ���Ѿ�д����ļ����� 
	ifstream INPUT(a, ios::binary);
	if (INPUT.fail()) { f = 0; cout << "�Բ������Ľ�ѹ�ļ������ڣ�" << endl; return; }
	ofstream OUTPUT(b, ios::binary);
	if (OUTPUT.fail())return;
	INPUT.read((char*)&FileLength, sizeof(FileLength));                          //��ȡԴ�ļ����� 
	for (int i = 0; i < 256; i++)                                                      //��ȡƵ���������´��� 
	{
		INPUT.read((char*)&node[i], sizeof(node[i]));
	}
	priority_queue<Tree*, vector<Tree*>, cmp> forest;                             //���ȶ��н���Huffman�� 
	creatTree(forest);
	unsigned char inByte;
	Tree* p = forest.top();                                                       //ָ��ָ��Huffman�����ڵ�Ԥ������ 
	while (!INPUT.eof())
	{
		INPUT.read((char*)&inByte, sizeof(inByte));                           //�����ֽڶ�ȡ
		if (!INPUT.eof())
		{
			for (int i = 0; i < 8; i++)                                           //��λ�������� 
			{
				int temp = inByte & 128;                                      //��10000000�����������ȡ��һλ 
				if (temp == 128)                                             //��һλΪ1�������Һ��� 
				{
					p = p->right;
					if (p->left == NULL && p->right == NULL)                    //���������Ҷ�ӽڵ㣬��ȡ�ڵ��е�Դ��д���ļ� 
					{
						unsigned char outByte = (char)p->code;
						OUTPUT.write((char*)&outByte, sizeof(outByte));
						WriteLength++;
						if (WriteLength == FileLength)break;                  //δ���������ֽ�ʱ���Ѿ�д���ļ������д�� 
						p = forest.top();                                    //ָ��ָ�ظ��ڵ� 
					}
				}
				else                                                       //��һλΪ0���������� 
				{
					p = p->left;
					if (p->left == NULL && p->right == NULL)                    //���������Ҷ�ӽڵ㣬��ȡ�ڵ��е�Դ��д���ļ� 
					{
						unsigned char outByte = (char)p->code;
						OUTPUT.write((char*)&outByte, sizeof(outByte));
						WriteLength++;
						if (WriteLength == FileLength)break;
						p = forest.top();                                    //ָ��ָ�ظ��ڵ� 
					}
				}
				inByte <<= 1;                                                 //����λ 
			}
		}
	}
	INPUT.close();
	OUTPUT.close();

}
int main()
{
	clock_t start, end;
	while (1)
	{
		system("cls");
		cout << "ʹ��˵��:" << endl;
		cout << endl;
		cout << "��1��ѹ��->��2����ѹ->��0���˳�" << endl;
		char a = _getch();
		if (a == '1')
		{
			cout << "����ǰΪѹ��ģʽ��" << endl;
			memset(node, 0, sizeof(node));
			cout << "�������ѹ���ļ�·��:";
			char inputFile[200];
			cin.getline(inputFile, 200);
			cout << "������Ŀ��·��:";
			char outputFile[200];
			cin.getline(outputFile, 200);
			cout << "ѹ����,�����ĵȴ�..." << endl;
			start = clock();
			int flag = 1;
			compress(inputFile, outputFile, flag);
			end = clock();
			double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
			if (flag == 1)cout << "��ϲѹ���ɹ���" << "ѹ����ʱ��" << cpu_time_used << endl;
		}
		else if (a == '2')
		{
			cout << "����ǰΪ��ѹ��ģʽ��" << endl;
			memset(node, 0, sizeof(node));
			cout << "���������ѹ�ļ�·��:";
			char outputFile1[200];
			cin.getline(outputFile1, 200);
			cout << "������Ŀ��·��:";
			char outbackFile2[200];
			cin.getline(outbackFile2, 200);
			cout << "��ѹ��,�����ĵȴ�..." << endl;
			start = clock();
			int flag = 1;
			uncompress(outputFile1, outbackFile2, flag);
			end = clock();
			double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
			if (flag == 1)cout << "��ϲ��ѹ�ɹ���" << "��ѹ��ʱ��" << cpu_time_used << endl;
		}
		else break;
		system("pause");

	}
	return 0;
}
#endif