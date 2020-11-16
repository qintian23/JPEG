#if 0
#include<iostream>
#include<conio.h> 
#include<string>
#include<vector>
#include<fstream>
#include<queue>
#include<ctime> 
using namespace std;
long node[256];//用于存储ASCII码在文件中出现的频率 
string m[256];//Huffman编码与ASCII匹配数组       
long FileLength = 0;//统计文件长度处理文件结尾字节匹配问题 
/*
Huffman树节点的建立：
weight:节点的权重  code:该节点代表的源码，left和right:左右孩子
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
Huffman树建立的优先队列比较结构体
用于构建优先队列的排序顺序（按照频率由小到大排列）
*/
struct cmp
{
	bool operator()(Tree* a, Tree* b)
	{
		return a->weight > b->weight;
	}
};
//统计ASCII码各字符在文件中出现的频率函数 
void statistics(char* a)
{
	FileLength = 0;
	ifstream INPUT(a, ios::binary);                                              //打开文件 
	if (INPUT.fail())return;                                                     //如果文件不存在则返回 
	while (!INPUT.eof())                                                         //循环读取每一字节，存入temp中并统计频率 
	{
		unsigned char temp;
		INPUT.read((char*)&temp, sizeof(unsigned char));
		int aa = (int)temp;
		if (!INPUT.eof()) { node[(int)temp]++; FileLength++; }                   //如果未读至文件尾部则加对应码频率，同时文件长度统计加一 
	}
	INPUT.close();                                                              //关闭文件 
}
//Huffman编码函数 
void encode(Tree* root, string a)                                                //对树进行递归遍历编码 
{
	if (root->left == NULL && root->right == NULL) 
	{ 
		m[root->code] = a; 
	}                //如果搜索至叶子节点，将匹配信息存入m中 
	else {
		if (root->left != NULL) {                                            //搜索左孩子并把字符串加'0’ 
			a += '0';  encode(root->left, a);
		}
		if (root->right != NULL) {                                           //搜索右孩子并把字符串加'1' 
			a = a.substr(0, a.length() - 1); a += '1';  encode(root->right, a);
		}
	}
}
//建立Huffman树函数 
void creatTree(priority_queue<Tree*, vector<Tree*>, cmp>& forest)
{
	for (int i = 0; i < 256; i++)
	{
		if (node[i] != 0)                                                    //将非0的Huffman编码统计数据以节点的形式加入到队列中 
		{
			Tree* temp = new Tree(node[i], i, NULL, NULL);                   //建立新节点 
			forest.push(temp);                                          //加入队列中 
		}
	}
	while (1)
	{
		if (forest.size() == 1) 
			break;                                      //始终循环直到队列中剩余一个节点为止，建树完成 
		Tree* ltemp = forest.top();                                        //将优先队列的队前两元素取出并pop掉，将两个节点作为新节点的左右孩子 
		forest.pop();
		Tree* rtemp = forest.top();
		forest.pop();
		Tree* temp = new Tree(ltemp->weight + rtemp->weight, 0, ltemp, rtemp);  //新节点的频率为两个孩子节点的频率之和，新节点不需要代表任何源码 
		forest.push(temp);

	}
}
/*压缩函数
INPUT:打开源文件  OUTPUT:打开压缩文件
inByte：临时存储的读入字节 outByte：临时存储的写入字节
storage：等待被写入的信息  f:判断是否进行了压缩
大致流程为：统计频率、建立Huffman树、对Huffman树进行编码、重读文件依次对应写入新编码及最后补位
*/
void compress(char* a, char* b, int& f)
{
	ifstream INPUT(a, ios::binary);                                             //打开源文件 
	if (INPUT.fail()) { f = 0; cout << "对不起，您的压缩文件不存在！" << endl; return; }
	ofstream OUTPUT(b, ios::binary);                                            //打开压缩文件预备写入 
	if (OUTPUT.fail())return;
	statistics(a);                                                             //统计字符出现的频率及文件长度 
	if (FileLength == 0 || FileLength == 1) { f = 0; cout << "您的文件为空文件，无需压缩！" << endl; return; }
	else if (FileLength <= 10) { cout << "友情提示：您的文件为小文件，压缩效果不会很理想！" << endl; }
	priority_queue<Tree*, vector<Tree*>, cmp> forest;
	creatTree(forest);                                                         //优先队列建立Huffman树
	string str = "";
	encode(forest.top(), str);                                                  //根据建树对Huffman树进行Huffman编码 
	OUTPUT.write((char*)&FileLength, sizeof(FileLength));                       //写入统计好的文件长度写入 
	for (int i = 0; i < 256; i++)                                                     //首先将统计频率信息写入文件，以便解压使用 
	{
		OUTPUT.write((char*)&node[i], sizeof(node[i]));
	}
	string storage = "";
	unsigned char inByte, outByte;
	while (!INPUT.eof()) {
		while (storage.length() < 8)                                       //扫描文件增添storage，存储信息超过八位跳出循环执行写入 
		{
			if (INPUT.eof())break;
			INPUT.read((char*)&inByte, sizeof(inByte));
			if (!INPUT.eof())storage += m[(int)inByte];                   //增加准备写入的Huffman编码                   
		}
		while (storage.length() >= 8)                                      //写入编码 
		{
			outByte = '\0';                                            //初始化写入字节 
			for (int i = 0; i < 8; i++)                                    //按位存入字节信息 
			{
				outByte <<= 1;                                          //左移字节 
				if (storage[i] == '1') outByte |= 1;                       //或操作写入1 
			}
			OUTPUT.write((char*)&outByte, sizeof(outByte));           //写入字节 
			storage = storage.substr(8);                                //写入8位后截掉storage的前8位 
		}
		if (INPUT.eof() && storage.length() != 0)                            //如果读取至文件结尾后还未完成编码写入 
		{
			outByte = '\0';
			for (int i = 0; i < storage.length(); i++)                    //剩余的位数写入 
			{
				outByte <<= 1;
				if (storage[i] == '1')outByte |= 1;
			}
			outByte <<= (8 - storage.length());                         //补足8位操作 
			OUTPUT.write((char*)&outByte, sizeof(outByte));         //写入最后一位字节 
		}
	}
	INPUT.close();
	OUTPUT.close();

}
/*解压函数
INPUT:打开压缩文件  OUTPUT:打开解压文件
inByte：临时存储的读入字节   outByte: 临时存储的写入字节
p：遍历Huffman树需要的指针   f:判断是否进行了解压
大致流程为：读取频率，重新建树，索引源码，写入文件
*/
void uncompress(char* a, char* b, int& f)
{
	long long WriteLength = 0;                                                    //统计已经写入的文件长度 
	ifstream INPUT(a, ios::binary);
	if (INPUT.fail()) { f = 0; cout << "对不起，您的解压文件不存在！" << endl; return; }
	ofstream OUTPUT(b, ios::binary);
	if (OUTPUT.fail())return;
	INPUT.read((char*)&FileLength, sizeof(FileLength));                          //读取源文件长度 
	for (int i = 0; i < 256; i++)                                                      //读取频率数组重新存入 
	{
		INPUT.read((char*)&node[i], sizeof(node[i]));
	}
	priority_queue<Tree*, vector<Tree*>, cmp> forest;                             //优先队列建立Huffman树 
	creatTree(forest);
	unsigned char inByte;
	Tree* p = forest.top();                                                       //指针指向Huffman树根节点预备遍历 
	while (!INPUT.eof())
	{
		INPUT.read((char*)&inByte, sizeof(inByte));                           //按照字节读取
		if (!INPUT.eof())
		{
			for (int i = 0; i < 8; i++)                                           //按位解析处理 
			{
				int temp = inByte & 128;                                      //与10000000进行与操作提取第一位 
				if (temp == 128)                                             //第一位为1，遍历右孩子 
				{
					p = p->right;
					if (p->left == NULL && p->right == NULL)                    //如果遍历至叶子节点，读取节点中的源码写入文件 
					{
						unsigned char outByte = (char)p->code;
						OUTPUT.write((char*)&outByte, sizeof(outByte));
						WriteLength++;
						if (WriteLength == FileLength)break;                  //未读完最后的字节时若已经写毕文件则完成写入 
						p = forest.top();                                    //指针指回根节点 
					}
				}
				else                                                       //第一位为0，遍历左孩子 
				{
					p = p->left;
					if (p->left == NULL && p->right == NULL)                    //如果遍历至叶子节点，读取节点中的源码写入文件 
					{
						unsigned char outByte = (char)p->code;
						OUTPUT.write((char*)&outByte, sizeof(outByte));
						WriteLength++;
						if (WriteLength == FileLength)break;
						p = forest.top();                                    //指针指回根节点 
					}
				}
				inByte <<= 1;                                                 //左移位 
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
		cout << "使用说明:" << endl;
		cout << endl;
		cout << "【1】压缩->【2】解压->【0】退出" << endl;
		char a = _getch();
		if (a == '1')
		{
			cout << "【当前为压缩模式】" << endl;
			memset(node, 0, sizeof(node));
			cout << "请输入待压缩文件路径:";
			char inputFile[200];
			cin.getline(inputFile, 200);
			cout << "请输入目的路径:";
			char outputFile[200];
			cin.getline(outputFile, 200);
			cout << "压缩中,请耐心等待..." << endl;
			start = clock();
			int flag = 1;
			compress(inputFile, outputFile, flag);
			end = clock();
			double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
			if (flag == 1)cout << "恭喜压缩成功！" << "压缩用时：" << cpu_time_used << endl;
		}
		else if (a == '2')
		{
			cout << "【当前为解压缩模式】" << endl;
			memset(node, 0, sizeof(node));
			cout << "请输入待解压文件路径:";
			char outputFile1[200];
			cin.getline(outputFile1, 200);
			cout << "请输入目的路径:";
			char outbackFile2[200];
			cin.getline(outbackFile2, 200);
			cout << "解压中,请耐心等待..." << endl;
			start = clock();
			int flag = 1;
			uncompress(outputFile1, outbackFile2, flag);
			end = clock();
			double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
			if (flag == 1)cout << "恭喜解压成功！" << "解压用时：" << cpu_time_used << endl;
		}
		else break;
		system("pause");

	}
	return 0;
}
#endif