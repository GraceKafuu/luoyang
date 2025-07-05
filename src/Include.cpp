#include "Include.h"

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

vector<string> stringSplit(string &str, string delim)
{
    regex reg(delim);
    vector<string> elems(sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                   sregex_token_iterator());
    return elems;
}

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
