#include "Include.h"

/**
 * @brief 读取文件中的所有行
 * 
 * 打开指定文件，逐行读取文件内容并存储在向量中返回。
 * 主要用于读取配置文件和类别名称文件。
 * 
 * @param fileName 文件路径
 * @return vector<string> 文件中每行内容组成的向量
 */
vector<string> readLines(string &fileName)
{
    ifstream ifs;
    vector<string> lines;
    ifs.open(fileName, ios::in);
    string buf;
    while (getline(ifs, buf))
        lines.push_back(buf);
    ifs.close();
    return lines;
}

/**
 * @brief 字符串分割函数
 * 
 * 使用正则表达式作为分隔符对字符串进行分割。
 * 主要用于解析配置文件中的键值对。
 * 
 * @param str 待分割的字符串
 * @param delim 分割符（正则表达式）
 * @return vector<string> 分割后的字符串向量
 */
vector<string> stringSplit(string &str, string delim)
{
    regex reg(delim);
    vector<string> elems(sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                   sregex_token_iterator());
    return elems;
}

/**
 * @brief 读取键值对配置文件
 * 
 * 读取格式为"key=value"的配置文件，将其转换为键值对映射。
 * 先按行读取文件内容，再按等号分割每行内容，构建键值对映射。
 * 
 * @param fileName 配置文件路径
 * @return unordered_map<string, string> 键值对映射
 */
unordered_map<string, string> readMap(string &fileName)
{
    unordered_map<string, string> dict;

    vector<string> lines = readLines(fileName);
    for (string line : lines)
    {
        vector<string> elems = stringSplit(line, "=");
        dict[elems.at(0)] = elems.at(1);
    }
    return dict;
}