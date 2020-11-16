#if 1
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
double AriCode(vector<string>& seq, vector<string>& symbols, vector<double>& pros)
{
	int length = pros.size();
	vector<double> pro_sum(length, pros[0]);
	double code;
	for (int i = 1; i < pros.size(); i++)
	{
		pro_sum[i] = pro_sum[i - 1] + pros[i];
	}
	double left_border = 0.0;
	double right_border = 1.0;
	for (string item : seq)
	{
		double margin = right_border - left_border;
		int index_tmp = std::distance(begin(symbols), find(symbols.begin(), symbols.end(), item));
		double pro_tmp = pros[index_tmp];
		if (index_tmp == 0)
		{
			right_border = left_border + margin * pro_tmp;
		}
		else if (index_tmp == pros.size())
		{
			left_border = left_border + pro_sum[index_tmp - 1] * margin;
		}
		else
		{
			left_border = left_border + pro_sum[index_tmp - 1] * margin;
			right_border = left_border + margin * pro_tmp;
		}
	}
	code = left_border + 0.3 * (right_border - left_border);
	return code;
}
vector<string> AriDecode(vector<double>& pros, vector<string>& symbols, double code, int length_code)
{
	int length = pros.size();
	vector<double> pro_sum(length + 1, 0);
	vector<string> seq_recons(length_code);
	for (int i = 1; i < pros.size() + 1; i++)
	{
		pro_sum[i] = pro_sum[i - 1] + pros[i - 1];
	}
	double left_border = 0.0;
	double right_border = 1.0;
	for (int i = 0; i < length_code; i++)
	{
		int flag = -1;
		for (int j = 0; j < length; j++)
		{
			double margin = right_border - left_border;
			double tag = (code - left_border) / margin;
			if (tag > pro_sum[j] && tag < pro_sum[j + 1])
			{
				flag = j;
				right_border = left_border + pro_sum[j + 1] * margin;
				left_border = left_border + pro_sum[j] * margin;
				seq_recons[i] = symbols[j];
				break;
			}
		}
	}
	return seq_recons;
}
int main()
{
	float a = 0.2f;
	vector<string> seq{ "a3","a2","a4","a3","a4","a1" };
	vector<string> symbols{ "a1","a2","a3","a4" };
	vector<string> symbols_recons(seq.size());
	vector<double> pros{ 0.2,0.3,0.1,0.4 };
	double code = AriCode(seq, symbols, pros);
	cout << code << endl;;
	symbols_recons = AriDecode(pros, symbols, code, seq.size());
	for (auto item : symbols_recons)
		cout << item << " ";
}
#endif 