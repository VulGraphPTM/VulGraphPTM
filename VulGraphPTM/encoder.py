import os
import json
import clang.cindex
import clang.enumerations
import csv
import numpy as np
import os
import re 
import warnings
import re
import nltk
import torch
import shutil
import uuid
import subprocess
warnings.filterwarnings('ignore')
# set the config
try:
    clang.cindex.Config.set_library_path("/usr/lib/x86_64-linux-gnu")
    clang.cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-10.so.1')
except:
    pass
from transformers import AutoTokenizer, AutoModel
from graphviz import Digraph
from gensim.models import Word2Vec
from tqdm import tqdm

warnings.filterwarnings('ignore')

def main():

    # 加载 CodeBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # 文本示例
    text = """
FFTContext *av_fft_init(int nbits, int inverse) {
    int a[60];
    FFTContext *s = av_malloc(sizeof(*s)) ;
    if (s && ff_fft_init(s, nbits , inverse))
        av_freep(&s) ;
    return s;
}
"""

    # Tokenize 文本并返回 token 和 token 类型 ID
    tokenized_input = tokenizer(text, return_tensors="pt", return_token_type_ids=True)

    # 获取 token 的编码向量
    input_ids = tokenized_input["input_ids"]
    attention_mask = tokenized_input["attention_mask"]
    token_type_ids = tokenized_input["token_type_ids"]

    # 输出 token 和 token 类型
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_types = [tokenizer.decode(token_type_id) for token_type_id in token_type_ids[0]]

    print("Tokens:", tokens)
    print("Token Types:", token_types)


keywords = ["alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
            "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
            "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
            "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
            "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
            "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
            "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
            "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr",
            "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
            "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this",
            "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned",
            "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "NULL"]
puncs = '~`!@#$%^&*()-+={[]}|\\;:\'\"<,>.?/'
puncs = list(puncs)

l_funcs = ['StrNCat', 'getaddrinfo', '_ui64toa', 'fclose', 'pthread_mutex_lock', 'gets_s', 'sleep', 
           '_ui64tot', 'freopen_s', '_ui64tow', 'send', 'lstrcat', 'HMAC_Update', '__fxstat', 'StrCatBuff', 
           '_mbscat', '_mbstok_s', '_cprintf_s', 'ldap_search_init_page', 'memmove_s', 'ctime_s', 'vswprintf', 
           'vswprintf_s', '_snwprintf', '_gmtime_s', '_tccpy', '*RC6*', '_mbslwr_s', 'random', 
           '__wcstof_internal', '_wcslwr_s', '_ctime32_s', 'wcsncat*', 'MD5_Init', '_ultoa', 
           'snprintf', 'memset', 'syslog', '_vsnprintf_s', 'HeapAlloc', 'pthread_mutex_destroy', 
           'ChangeWindowMessageFilter', '_ultot', 'crypt_r', '_strupr_s_l', 'LoadLibraryExA', '_strerror_s', 
           'LoadLibraryExW', 'wvsprintf', 'MoveFileEx', '_strdate_s', 'SHA1', 'sprintfW', 'StrCatNW', 
           '_scanf_s_l', 'pthread_attr_init', '_wtmpnam_s', 'snscanf', '_sprintf_s_l', 'dlopen', 
           'sprintfA', 'timed_mutex', 'OemToCharA', 'ldap_delete_ext', 'sethostid', 'popen', 'OemToCharW', 
           '_gettws', 'vfork', '_wcsnset_s_l', 'sendmsg', '_mbsncat', 'wvnsprintfA', 'HeapFree', '_wcserror_s', 
           'realloc', '_snprintf*', 'wcstok', '_strncat*', 'StrNCpy', '_wasctime_s', 'push*', '_lfind_s', 
           'CC_SHA512', 'ldap_compare_ext_s', 'wcscat_s', 'strdup', '_chsize_s', 'sprintf_s', 'CC_MD4_Init', 
           'wcsncpy', '_wfreopen_s', '_wcsupr_s', '_searchenv_s', 'ldap_modify_ext_s', '_wsplitpath', 
           'CC_SHA384_Final', 'MD2', 'RtlCopyMemory', 'lstrcatW', 'MD4', 'MD5', '_wcstok_s_l', '_vsnwprintf_s', 
           'ldap_modify_s', 'strerror', '_lsearch_s', '_mbsnbcat_s', '_wsplitpath_s', 'MD4_Update', '_mbccpy_s', 
           '_strncpy_s_l', '_snprintf_s', 'CC_SHA512_Init', 'fwscanf_s', '_snwprintf_s', 'CC_SHA1', 'swprintf', 
           'fprintf', 'EVP_DigestInit_ex', 'strlen', 'SHA1_Init', 'strncat', '_getws_s', 'CC_MD4_Final', 
           'wnsprintfW', 'lcong48', 'lrand48', 'write', 'HMAC_Init', '_wfopen_s', 'wmemchr', '_tmakepath', 
           'wnsprintfA', 'lstrcpynW', 'scanf_s', '_mbsncpy_s_l', '_localtime64_s', 'fstream.open', '_wmakepath', 
           'Connection.open', '_tccat', 'valloc', 'setgroups', 'unlink', 'fstream.put', 'wsprintfA', '*SHA1*', 
           '_wsearchenv_s', 'ualstrcpyA', 'CC_MD5_Update', 'strerror_s', 'HeapCreate', 'ualstrcpyW', '__xstat', 
           '_wmktemp_s', 'StrCatChainW', 'ldap_search_st', '_mbstowcs_s_l', 'ldap_modify_ext', '_mbsset_s', 
           'strncpy_s', 'move', 'execle', 'StrCat', 'xrealloc', 'wcsncpy_s', '_tcsncpy*', 'execlp', 
           'RIPEMD160_Final', 'ldap_search_s', 'EnterCriticalSection', '_wctomb_s_l', 'fwrite', '_gmtime64_s', 
           'sscanf_s', 'wcscat', '_strupr_s', 'wcrtomb_s', 'VirtualLock', 'ldap_add_ext_s', '_mbscpy', 
           '_localtime32_s', 'lstrcpy', '_wcsncpy*', 'CC_SHA1_Init', '_getts', '_wfopen', '__xstat64', 
           'strcoll', '_fwscanf_s_l', '_mbslwr_s_l', 'RegOpenKey', 'makepath', 'seed48', 'CC_SHA256', 
           'sendto', 'execv', 'CalculateDigest', 'memchr', '_mbscpy_s', '_strtime_s', 'ldap_search_ext_s', 
           '_chmod', 'flock', '__fxstat64', '_vsntprintf', 'CC_SHA256_Init', '_itoa_s', '__wcserror_s', 
           '_gcvt_s', 'fstream.write', 'sprintf', 'recursive_mutex', 'strrchr', 'gethostbyaddr', '_wcsupr_s_l', 
           'strcspn', 'MD5_Final', 'asprintf', '_wcstombs_s_l', '_tcstok', 'free', 'MD2_Final', 'asctime_s', 
           '_alloca', '_wputenv_s', '_wcsset_s', '_wcslwr_s_l', 'SHA1_Update', 'filebuf.sputc', 'filebuf.sputn', 
           'SQLConnect', 'ldap_compare', 'mbstowcs_s', 'HMAC_Final', 'pthread_condattr_init', '_ultow_s', 'rand', 
           'ofstream.put', 'CC_SHA224_Final', 'lstrcpynA', 'bcopy', 'system', 'CreateFile*', 'wcscpy_s', 
           '_mbsnbcpy*', 'open', '_vsnwprintf', 'strncpy', 'getopt_long', 'CC_SHA512_Final', '_vsprintf_s_l', 
           'scanf', 'mkdir', '_localtime_s', '_snprintf', '_mbccpy_s_l', 'memcmp', 'final', '_ultoa_s', 
           'lstrcpyW', 'LoadModule', '_swprintf_s_l', 'MD5_Update', '_mbsnset_s_l', '_wstrtime_s', '_strnset_s', 
           'lstrcpyA', '_mbsnbcpy_s', 'mlock', 'IsBadHugeWritePtr', 'copy', '_mbsnbcpy_s_l', 'wnsprintf', 
           'wcscpy', 'ShellExecute', 'CC_MD4', '_ultow', '_vsnwprintf_s_l', 'lstrcpyn', 'CC_SHA1_Final', 
           'vsnprintf', '_mbsnbset_s', '_i64tow', 'SHA256_Init', 'wvnsprintf', 'RegCreateKey', 'strtok_s', 
           '_wctime32_s', '_i64toa', 'CC_MD5_Final', 'wmemcpy', 'WinExec', 'CreateDirectory*', 
           'CC_SHA256_Update', '_vsnprintf_s_l', 'jrand48', 'wsprintf', 'ldap_rename_ext_s', 'filebuf.open', 
           '_wsystem', 'SHA256_Update', '_cwscanf_s', 'wsprintfW', '_sntscanf', '_splitpath', 'fscanf_s', 
           'strpbrk', 'wcstombs_s', 'wscanf', '_mbsnbcat_s_l', 'strcpynA', 'pthread_cond_init', 'wcsrtombs_s', 
           '_wsopen_s', 'CharToOemBuffA', 'RIPEMD160_Update', '_tscanf', 'HMAC', 'StrCCpy', 'Connection.connect', 
           'lstrcatn', '_mbstok', '_mbsncpy', 'CC_SHA384_Update', 'create_directories', 'pthread_mutex_unlock', 
           'CFile.Open', 'connect', '_vswprintf_s_l', '_snscanf_s_l', 'fputc', '_wscanf_s', '_snprintf_s_l', 
           'strtok', '_strtok_s_l', 'lstrcatA', 'snwscanf', 'pthread_mutex_init', 'fputs', 'CC_SHA384_Init', 
           '_putenv_s', 'CharToOemBuffW', 'pthread_mutex_trylock', '__wcstoul_internal', '_memccpy', 
           '_snwprintf_s_l', '_strncpy*', 'wmemset', 'MD4_Init', '*RC4*', 'strcpyW', '_ecvt_s', 'memcpy_s', 
           'erand48', 'IsBadHugeReadPtr', 'strcpyA', 'HeapReAlloc', 'memcpy', 'ldap_rename_ext', 'fopen_s', 
           'srandom', '_cgetws_s', '_makepath', 'SHA256_Final', 'remove', '_mbsupr_s', 'pthread_mutexattr_init', 
           '__wcstold_internal', 'StrCpy', 'ldap_delete', 'wmemmove_s', '_mkdir', 'strcat', '_cscanf_s_l', 
           'StrCAdd', 'swprintf_s', '_strnset_s_l', 'close', 'ldap_delete_ext_s', 'ldap_modrdn', 'strchr', 
           '_gmtime32_s', '_ftcscat', 'lstrcatnA', '_tcsncat', 'OemToChar', 'mutex', 'CharToOem', 'strcpy_s', 
           'lstrcatnW', '_wscanf_s_l', '__lxstat64', 'memalign', 'MD2_Init', 'StrCatBuffW', 'StrCpyN', 'CC_MD5', 
           'StrCpyA', 'StrCatBuffA', 'StrCpyW', 'tmpnam_r', '_vsnprintf', 'strcatA', 'StrCpyNW', '_mbsnbset_s_l', 
           'EVP_DigestInit', '_stscanf', 'CC_MD2', '_tcscat', 'StrCpyNA', 'xmalloc', '_tcslen', '*MD4*', 
           'vasprintf', 'strxfrm', 'chmod', 'ldap_add_ext', 'alloca', '_snscanf_s', 'IsBadWritePtr', 'swscanf_s', 
           'wmemcpy_s', '_itoa', '_ui64toa_s', 'EVP_DigestUpdate', '__wcstol_internal', '_itow', 'StrNCatW', 
           'strncat_s', 'ualstrcpy', 'execvp', '_mbccat', 'EVP_MD_CTX_init', 'assert', 'ofstream.write', 
           'ldap_add', '_sscanf_s_l', 'drand48', 'CharToOemW', 'swscanf', '_itow_s', 'RIPEMD160_Init', 
           'CopyMemory', 'initstate', 'getpwuid', 'vsprintf', '_fcvt_s', 'CharToOemA', 'setuid', 'malloc', 
           'StrCatNA', 'strcat_s', 'srand', 'getwd', '_controlfp_s', 'olestrcpy', '__wcstod_internal', 
           '_mbsnbcat', 'lstrncat', 'des_*', 'CC_SHA224_Init', 'set*', 'vsprintf_s', 'SHA1_Final', '_umask_s', 
           'gets', 'setstate', 'wvsprintfW', 'LoadLibraryEx', 'ofstream.open', 'calloc', '_mbstrlen', 
           '_cgets_s', '_sopen_s', 'IsBadStringPtr', 'wcsncat_s', 'add*', 'nrand48', 'create_directory', 
           'ldap_search_ext', '_i64toa_s', '_ltoa_s', '_cwscanf_s_l', 'wmemcmp', '__lxstat', 'lstrlen', 
           'pthread_condattr_destroy', '_ftcscpy', 'wcstok_s', '__xmknod', 'pthread_attr_destroy', 'sethostname', 
           '_fscanf_s_l', 'StrCatN', 'RegEnumKey', '_tcsncpy', 'strcatW', 'AfxLoadLibrary', 'setenv', 'tmpnam', 
           '_mbsncat_s_l', '_wstrdate_s', '_wctime64_s', '_i64tow_s', 'CC_MD4_Update', 'ldap_add_s', '_umask', 
           'CC_SHA1_Update', '_wcsset_s_l', '_mbsupr_s_l', 'strstr', '_tsplitpath', 'memmove', '_tcscpy', 
           'vsnprintf_s', 'strcmp', 'wvnsprintfW', 'tmpfile', 'ldap_modify', '_mbsncat*', 'mrand48', 'sizeof', 
           'StrCatA', '_ltow_s', '*desencrypt*', 'StrCatW', '_mbccpy', 'CC_MD2_Init', 'RIPEMD160', 'ldap_search', 
           'CC_SHA224', 'mbsrtowcs_s', 'update', 'ldap_delete_s', 'getnameinfo', '*RC5*', '_wcsncat_s_l', 
           'DriverManager.getConnection', 'socket', '_cscanf_s', 'ldap_modrdn_s', '_wopen', 'CC_SHA256_Final', 
           '_snwprintf*', 'MD2_Update', 'strcpy', '_strncat_s_l', 'CC_MD5_Init', 'mbscpy', 'wmemmove', 
           'LoadLibraryW', '_mbslen', '*alloc', '_mbsncat_s', 'LoadLibraryA', 'fopen', 'StrLen', 'delete', 
           '_splitpath_s', 'CreateFileTransacted*', 'MD4_Final', '_open', 'CC_SHA384', 'wcslen', 'wcsncat', 
           '_mktemp_s', 'pthread_mutexattr_destroy', '_snwscanf_s', '_strset_s', '_wcsncpy_s_l', 'CC_MD2_Final', 
           '_mbstok_s_l', 'wctomb_s', 'MySQL_Driver.connect', '_snwscanf_s_l', '*_des_*', 'LoadLibrary', 
           '_swscanf_s_l', 'ldap_compare_s', 'ldap_compare_ext', '_strlwr_s', 'GetEnvironmentVariable', 
           'cuserid', '_mbscat_s', 'strspn', '_mbsncpy_s', 'ldap_modrdn2', 'LeaveCriticalSection', 'CopyFile', 
           'getpwd', 'sscanf', 'creat', 'RegSetValue', 'ldap_modrdn2_s', 'CFile.Close', '*SHA_1*', 
           'pthread_cond_destroy', 'CC_SHA512_Update', '*RC2*', 'StrNCatA', '_mbsnbcpy', '_mbsnset_s', 
           'crypt', 'excel', '_vstprintf', 'xstrdup', 'wvsprintfA', 'getopt', 'mkstemp', '_wcsnset_s', 
           '_stprintf', '_sntprintf', 'tmpfile_s', 'OpenDocumentFile', '_mbsset_s_l', '_strset_s_l', 
           '_strlwr_s_l', 'ifstream.open', 'xcalloc', 'StrNCpyA', '_wctime_s', 'CC_SHA224_Update', '_ctime64_s', 
           'MoveFile', 'chown', 'StrNCpyW', 'IsBadReadPtr', '_ui64tow_s', 'IsBadCodePtr', 'getc', 
           'OracleCommand.ExecuteOracleScalar', 'AccessDataSource.Insert', 'IDbDataAdapter.FillSchema', 
           'IDbDataAdapter.Update', 'GetWindowText*', 'SendMessage', 'SqlCommand.ExecuteNonQuery', 'streambuf.sgetc', 
           'streambuf.sgetn', 'OracleCommand.ExecuteScalar', 'SqlDataSource.Update', '_Read_s', 'IDataAdapter.Fill', 
           '_wgetenv', '_RecordsetPtr.Open*', 'AccessDataSource.Delete', 'Recordset.Open*', 'filebuf.sbumpc', 'DDX_*', 
           'RegGetValue', 'fstream.read*', 'SqlCeCommand.ExecuteResultSet', 'SqlCommand.ExecuteXmlReader', 'main', 
           'streambuf.sputbackc', 'read', 'm_lpCmdLine', 'CRichEditCtrl.Get*', 'istream.putback', 
           'SqlCeCommand.ExecuteXmlReader', 'SqlCeCommand.BeginExecuteXmlReader', 'filebuf.sgetn', 
           'OdbcDataAdapter.Update', 'filebuf.sgetc', 'SQLPutData', 'recvfrom', 'OleDbDataAdapter.FillSchema', 
           'IDataAdapter.FillSchema', 'CRichEditCtrl.GetLine', 'DbDataAdapter.Update', 'SqlCommand.ExecuteReader', 
           'istream.get', 'ReceiveFrom', '_main', 'fgetc', 'DbDataAdapter.FillSchema', 'kbhit', 'UpdateCommand.Execute*', 
           'Statement.execute', 'fgets', 'SelectCommand.Execute*', 'getch', 'OdbcCommand.ExecuteNonQuery', 
           'CDaoQueryDef.Execute', 'fstream.getline', 'ifstream.getline', 'SqlDataAdapter.FillSchema', 
           'OleDbCommand.ExecuteReader', 'Statement.execute*', 'SqlCeCommand.BeginExecuteNonQuery', 
           'OdbcCommand.ExecuteScalar', 'SqlCeDataAdapter.Update', 'sendmessage', 'mysqlpp.DBDriver', 'fstream.peek', 
           'Receive', 'CDaoRecordset.Open', 'OdbcDataAdapter.FillSchema', '_wgetenv_s', 'OleDbDataAdapter.Update', 
           'readsome', 'SqlCommand.BeginExecuteXmlReader', 'recv', 'ifstream.peek', '_Main', '_tmain', '_Readsome_s', 
           'SqlCeCommand.ExecuteReader', 'OleDbCommand.ExecuteNonQuery', 'fstream.get', 'IDbCommand.ExecuteScalar', 
           'filebuf.sputbackc', 'IDataAdapter.Update', 'streambuf.sbumpc', 'InsertCommand.Execute*', 'RegQueryValue', 
           'IDbCommand.ExecuteReader', 'SqlPipe.ExecuteAndSend', 'Connection.Execute*', 'getdlgtext', 'ReceiveFromEx', 
           'SqlDataAdapter.Update', 'RegQueryValueEx', 'SQLExecute', 'pread', 'SqlCommand.BeginExecuteReader', 'AfxWinMain', 
           'getchar', 'istream.getline', 'SqlCeDataAdapter.Fill', 'OleDbDataReader.ExecuteReader', 'SqlDataSource.Insert', 
           'istream.peek', 'SendMessageCallback', 'ifstream.read*', 'SqlDataSource.Select', 'SqlCommand.ExecuteScalar', 
           'SqlDataAdapter.Fill', 'SqlCommand.BeginExecuteNonQuery', 'getche', 'SqlCeCommand.BeginExecuteReader', 'getenv', 
           'streambuf.snextc', 'Command.Execute*', '_CommandPtr.Execute*', 'SendNotifyMessage', 'OdbcDataAdapter.Fill', 
           'AccessDataSource.Update', 'fscanf', 'QSqlQuery.execBatch', 'DbDataAdapter.Fill', 'cin', 
           'DeleteCommand.Execute*', 'QSqlQuery.exec', 'PostMessage', 'ifstream.get', 'filebuf.snextc', 
           'IDbCommand.ExecuteNonQuery', 'Winmain', 'fread', 'getpass', 'GetDlgItemTextCCheckListBox.GetCheck', 
           'DISP_PROPERTY_EX', 'pread64', 'Socket.Receive*', 'SACommand.Execute*', 'SQLExecDirect', 
           'SqlCeDataAdapter.FillSchema', 'DISP_FUNCTION', 'OracleCommand.ExecuteNonQuery', 'CEdit.GetLine', 
           'OdbcCommand.ExecuteReader', 'CEdit.Get*', 'AccessDataSource.Select', 'OracleCommand.ExecuteReader', 
           'OCIStmtExecute', 'getenv_s', 'DB2Command.Execute*', 'OracleDataAdapter.FillSchema', 'OracleDataAdapter.Fill', 
           'CComboBox.Get*', 'SqlCeCommand.ExecuteNonQuery', 'OracleCommand.ExecuteOracleNonQuery', 'mysqlpp.Query', 
           'istream.read*', 'CListBox.GetText', 'SqlCeCommand.ExecuteScalar', 'ifstream.putback', 'readlink', 
           'CHtmlEditCtrl.GetDHtmlDocument', 'PostThreadMessage', 'CListCtrl.GetItemText', 'OracleDataAdapter.Update', 
           'OleDbCommand.ExecuteScalar', 'stdin', 'SqlDataSource.Delete', 'OleDbDataAdapter.Fill', 'fstream.putback', 
           'IDbDataAdapter.Fill', '_wspawnl', 'fwprintf', 'sem_wait', '_unlink', 'ldap_search_ext_sW', 'signal', 'PQclear', 
           'PQfinish', 'PQexec', 'PQresultStatus']

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data

def read_code_file(file_path):
    code_lines = {}
    with open(file_path) as fp:
        for ln, line in enumerate(fp):
            assert isinstance(line, str)
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')]
            code_lines[ln + 1] = line
        return code_lines

def extract_nodes_with_location_info(nodes):
    # Will return an array identifying the indices of those nodes in nodes array,
    # another array identifying the node_id of those nodes
    # another array indicating the line numbers
    # all 3 return arrays should have same length indicating 1-to-1 matching.
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number

def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True :#edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'CONTROLS': #Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES': # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list

def create_visual_graph(code, adjacency_list, file_name='test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(file_name, view=verbose)

def create_forward_slice(adjacency_list, line_no):
    sliced_lines = set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append(line_no)
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        adjacents = adjacency_list[cur]
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    sliced_lines = sorted(sliced_lines)
    return sliced_lines

def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    return cgraph

def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            igraph[node].add(ln)
    return igraph

def create_backward_slice(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph(adjacency_list)
    return create_forward_slice(inverted_adjacency_list, line_no)

class Tokenizer:
    # creates the object, does the inital parse
    def __init__(self, path, tokenizer_type='original'):
        self.index = clang.cindex.Index.create()
        self.tu = self.index.parse(path)
        self.path = self.extract_path(path)
        self.symbol_table = {}
        self.symbol_count = 1
        self.tokenizer_type = tokenizer_type

    # To output for split_functions, must have same path up to last two folders
    def extract_path(self, path):
        return "".join(path.split("/")[:-2])

    
    def full_tokenize_cursor(self, cursor):
        tokens = cursor.get_tokens()
        result = []
        for token in tokens:
            if token.kind.name == "COMMENT":
                continue
            if token.kind.name == "LITERAL":
                result += self.process_literal(token)
                continue
            if token.kind.name == "IDENTIFIER":
                result += ["ID"]
                continue
            result += [token.spelling]
        return result

    def full_tokenize(self):
        cursor = self.tu.cursor
        return self.full_tokenize_cursor(cursor)

    def process_literal(self, literal):
        cursor_kind = clang.cindex.CursorKind
        kind = literal.cursor.kind
        if kind == cursor_kind.INTEGER_LITERAL:
            return literal.spelling
        if kind == cursor_kind.FLOATING_LITERAL:
            return literal.spelling
        if kind == cursor_kind.IMAGINARY_LITERAL:
            return ["NUM"]       
        if kind == cursor_kind.STRING_LITERAL:
            return ["STRING"]
        sp = literal.spelling
        if re.match('[0-9]+', sp) is not None:
            return sp
        return ["LITERAL"]

    def split_functions(self, method_only):
        results = []
        cursor_kind = clang.cindex.CursorKind
        cursor = self.tu.cursor
        for c in cursor.get_children():
            filename = c.location.file.name if c.location.file != None else "NONE"
            extracted_path = self.extract_path(filename)

            if (c.kind == cursor_kind.CXX_METHOD or (method_only == False and c.kind == cursor_kind.FUNCTION_DECL)) and extracted_path == self.path:
                name = c.spelling
                tokens = self.full_tokenize_cursor(c)
                filename = filename.split("/")[-1]
                results += [tokens]

        return results


def tokenize(file_text):
    try:
        c_file = open('/tmp/test1.c', 'w')
        c_file.write(file_text)
        c_file.close()
        tok = Tokenizer('/tmp/test1.c')
        results = tok.split_functions(False)
        return ' '.join(results[0])
    except:
        return None

def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        return ' '.join(lines)
    
def extract_line_number(idx, nodes):
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except:
                    pass
        idx -= 1
    return -1

def func_extract_slices(split_dir, parsed):
    all_data = []
    files = []
    files += os.listdir(split_dir)
    print(len(files))
        
    for i, file_name  in enumerate(files):
        label = file_name.strip()[:-2].split('_')[-1]
        code_text = read_file(split_dir + file_name.strip())
        
        nodes_file_path = parsed + file_name.strip() + '/nodes.csv'
        edges_file_path = parsed + file_name.strip() + '/edges.csv'
        nc = open(nodes_file_path)
        nodes_file = csv.DictReader(nc, delimiter='\t')
        try:
            nodes = [node for node in nodes_file]
        except csv.Error as e:
            print(e)
            continue
        call_lines = set()
        array_lines = set()
        ptr_lines = set()
        arithmatic_lines = set()
        
        if len(nodes) == 0:
            continue
        
        for node_idx, node in enumerate(nodes):
            ntype = node['type'].strip()
            if ntype == 'CallExpression':
                function_name = nodes[node_idx + 1]['code']
                if function_name  is None or function_name.strip() == '':
                    continue
                if function_name.strip() in l_funcs:
                    line_no = extract_line_number(node_idx, nodes)
                    if line_no > 0:
                        call_lines.add(line_no)
            elif ntype == 'ArrayIndexing':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    array_lines.add(line_no)
            elif ntype == 'PtrMemberAccess':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    ptr_lines.add(line_no)
            elif node['operator'].strip() in ['+', '-', '*', '/']:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    arithmatic_lines.add(line_no)
            
        nodes = read_csv(nodes_file_path)
        edges = read_csv(edges_file_path)
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
        combined_graph = combine_control_and_data_adjacents(adjacency_list)
        
        array_slices = []
        array_slices_bdir = []
        call_slices = []
        call_slices_bdir = []
        arith_slices = []
        arith_slices_bdir = []
        ptr_slices = []
        ptr_slices_bdir = []
        all_slices = []
        
        
        all_keys = set()
        _keys = set()
        for slice_ln in call_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                call_slices.append(backward_sliced_lines)
                call_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
                
        _keys = set()
        for slice_ln in array_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                array_slices.append(backward_sliced_lines)
                array_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
        
        _keys = set()
        for slice_ln in arithmatic_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                arith_slices.append(backward_sliced_lines)
                arith_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
        
        _keys = set()
        for slice_ln in ptr_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                ptr_slices.append(backward_sliced_lines)
                ptr_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
                
        t_code = tokenize(code_text)
        if t_code is None:
            continue
        data_instance = {
            'file_path': split_dir + file_name.strip(),
            'code' : code_text,
            'tokenized': t_code,
            'call_slices_vd': call_slices,
            'call_slices_sy': call_slices_bdir,
            'array_slices_vd': array_slices,
            'array_slices_sy': array_slices_bdir,
            'arith_slices_vd': arith_slices,
            'arith_slices_sy': arith_slices_bdir,
            'ptr_slices_vd': ptr_slices,
            'ptr_slices_sy': ptr_slices_bdir,
            'label': int(label)
        }
        all_data.append(data_instance)
        
        if i % 1000 == 0:
            print(i, len(call_slices), len(call_slices_bdir), 
                len(array_slices), len(array_slices_bdir), 
                len(arith_slices), len(arith_slices_bdir), sep='\t')
    
    output_file = open('./tmp_full_data_with_slices.json', 'w')
    json.dump(all_data, output_file)
    output_file.close()


def symbolic_tokenize(code):
    tokens = nltk.word_tokenize(code)
    c_tokens = []
    for t in tokens:
        if t.strip() != '':
            c_tokens.append(t.strip())
    f_count = 1
    var_count = 1
    symbol_table = {}
    final_tokens = []
    for idx in range(len(c_tokens)):
        t = c_tokens[idx]
        if t in keywords:
            final_tokens.append(t)
        elif t in puncs:
            final_tokens.append(t)
        elif t in l_funcs:
            final_tokens.append(t)
        elif (idx+1) < len(c_tokens) and c_tokens[idx + 1] == '(':
            if t in keywords:
                final_tokens.append(t)
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t])
            idx += 1

        elif t.endswith('('):
            t = t[:-1]
            if t in keywords:
                final_tokens.append(t + '(')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '(')
        elif t.endswith('()'):
            t = t[:-2]
            if t in keywords:
                final_tokens.append(t + '( )')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '( )')
        elif re.match("^\"*\"$", t) is not None:
            final_tokens.append("STRING")
        elif re.match("^[0-9]+(\.[0-9]+)?$", t) is not None:
            final_tokens.append("NUMBER")
        elif re.match("^[0-9]*(\.[0-9]+)$", t) is not None:
            final_tokens.append("NUMBER")
        else:
            if t not in symbol_table.keys():
                symbol_table[t] = "VAR" + str(var_count)
                var_count += 1
            final_tokens.append(symbol_table[t])
    return ' '.join(final_tokens)

type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}
type_one_hot = np.eye(len(type_map))
# We currently consider 12 types of edges mentioned in ICST paper
edgeType_full = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}

# We currently consider 12 types of edges mentioned in ICST paper
edgeType_control = {
    'FLOWS_TO': 3,  # Control Flow
    'CONTROLS': 7,  # Control Dependency edge
}

edgeType_data = {
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
}

edgeType_control_data = {
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'FLOWS_TO': 3,  # Control Flow
    'CONTROLS': 7,  # Control Dependency edge
}

def code_to_tensor(code, wv=None, ptm_name=None):
    if wv:
        node_split = nltk.word_tokenize(code)
        nrp = np.zeros(100)
        for token in node_split:
            try:
                embedding = wv.wv[token]
            except:
                embedding = np.zeros(100)
            nrp = np.add(nrp, embedding)
        if len(node_split) > 0:
            fNrp = np.divide(nrp, len(node_split))
        else:
            fNrp = nrp
        return fNrp
    elif ptm_name:
        # inputs = tokenizer(code, padding=True, truncation=True, return_tensors="pt")
        # outputs = ptm_model(**inputs)
        # encoded_lines = outputs.last_hidden_state.mean(dim=1).tolist()
        # return encoded_lines
        pass

def inputGeneration(nodeCSV, edgeCSV, target, wv, edge_type_map, cfg_only=False, ptm_name=None):
    gInput = dict()
    gInput["targets"] = list()
    gInput["graph"] = list()
    gInput["node_line"] = list()
    gInput["node_code"] = list()
    gInput["node_features"] = list()
    gInput["targets"].append([target])
    with open(nodeCSV, 'r') as nc:
        nodes = csv.DictReader(nc, delimiter='\t')
        nodeMap = dict()
        allNodes = {}
        allCodes = {}
        allLines = {}
        node_idx = 0
        idx2node = {}
        line_list = []
        type_list = []
        for idx, node in enumerate(nodes):
            cfgNode = node['isCFGNode'].strip()
            if not cfg_only and (cfgNode == '' or cfgNode == 'False'):
                continue
            nodeKey = node['key']
            node_type = node['type']
            if node_type == 'File':
                continue
            node_content = node['code'].strip()
            line_list.append(node_content)
            type_list.append(node_type)
            location = node['location'] if 'location' in node.keys() else ''
            line_number = int(location.split(':')[0]) if location else -1
            node_split = nltk.word_tokenize(node_content)
            nrp = np.zeros(100)
            for token in node_split:
                try:
                    embedding = wv.wv[token]
                except:
                    embedding = np.zeros(100)
                nrp = np.add(nrp, embedding)
            if len(node_split) > 0:
                fNrp = np.divide(nrp, len(node_split))
            else:
                fNrp = nrp
            try:
                node_feature = type_one_hot[type_map[node_type] - 1].tolist()
            except:
                node_feature = [0] * len(type_map)
            node_feature.extend(fNrp.tolist())
            allNodes[nodeKey] = node_feature
            allLines[nodeKey] = line_number
            allCodes[nodeKey] = node_content
            nodeMap[nodeKey] = node_idx
            idx2node[node_idx] = nodeKey
            node_idx += 1
        # if ptm_name:
        #     encoded_lines = code_to_tensor(line_list, wv, ptm_name)
        #     idx = 0
        #     for ntype, fNrp in zip(type_list, encoded_lines):
        #         node_feature = type_one_hot[type_map[ntype] - 1].tolist()
        #         node_feature.extend(fNrp)
        #         allNodes[idx2node[idx]] = node_feature
        #         idx += 1
        if node_idx == 0 or node_idx >= 500:  # cut
            return None
        all_nodes_with_edges = set()
        trueNodeMap = {}
        all_edges = []
        with open(edgeCSV, 'r') as ec:
            reader = csv.DictReader(ec, delimiter='\t')
            for e in reader:
                start, end, eType = e["start"], e["end"], e["type"]
                if eType != "IS_FILE_OF":
                    if not start in nodeMap or not end in nodeMap or not eType in edge_type_map:
                        continue
                    all_nodes_with_edges.add(start)
                    all_nodes_with_edges.add(end)
                    edge = [start, edge_type_map[eType], end]
                    all_edges.append(edge)
        if len(all_edges) == 0:
            return None
        for i, node in enumerate(all_nodes_with_edges):
            trueNodeMap[node] = i
            gInput["node_features"].append(allNodes[node])
            gInput["node_line"].append(allLines[node])
            gInput["node_code"].append(allCodes[node])
        for edge in all_edges:
            start, t, end = edge
            start = trueNodeMap[start]
            end = trueNodeMap[end]
            e = [start, t, end]
            gInput["graph"].append(e)
    return gInput

def extract_slices(linized_code, list_of_slices):
    sliced_codes = []
    for slice in list_of_slices:
        tokenized = []
        for ln in slice:
            code = linized_code[ln]
            tokenized.append(symbolic_tokenize(code))
        sliced_codes.append(' '.join(tokenized))
    return sliced_codes

def unify_slices(list_of_list_of_slices):
    taken_slice = set()
    unique_slice_lines = []
    for list_of_slices in list_of_list_of_slices:
        for slice in list_of_slices:
            slice_id = str(slice)
            if slice_id not in taken_slice:
                unique_slice_lines.append(slice)
                taken_slice.add(slice_id)
    return unique_slice_lines


def func_create_ggnn_data(split_dir, parsed, wv_path=None, ptm_name=None):
    json_file_path = './tmp_full_data_with_slices.json'
    data = json.load(open(json_file_path))
    model = None
    if wv_path:
        model = Word2Vec.load(wv_path)
    final_data = []
    output_prefix = './tmp_dir/tmp_data.json'
    if os.path.exists('./tmp_dir'):
        shutil.rmtree('./tmp_dir')
    os.mkdir('./tmp_dir')
    v, nv, vd_present, syse_present, cg_present, dg_present, cdg_present = 0, 0, 0, 0, 0, 0, 0
    data_shard = 1
    for didx, entry in enumerate(tqdm(data)):
        file_name = entry['file_path'].split('/')[-1]
        nodes_path = os.path.join(parsed, file_name, 'nodes.csv')
        edges_path = os.path.join(parsed, file_name, 'edges.csv')
        label = int(entry['label'])
        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            continue
        linized_code = {}
        for ln, code in enumerate(entry['code'].split('\n')):
            linized_code[ln + 1] = code
        vuld_slices = extract_slices(linized_code, entry['call_slices_vd'])
        syse_slices = extract_slices(
            linized_code, unify_slices(
                [entry['call_slices_sy'], entry['array_slices_sy'], entry['arith_slices_sy'], entry['ptr_slices_sy']]
            )
        )
        graph_input_full = inputGeneration(
            nodes_path, edges_path, label, model, edgeType_full, False, ptm_name=ptm_name)
        # graph_input_control = inputGeneration(
        #     nodes_path, edges_path, label, model, edgeType_control, True, ptm_name=ptm_name)
        # graph_input_data = inputGeneration(nodes_path, edges_path, label, model, edgeType_data, True, ptm_name=ptm_name)
        # graph_input_cd = inputGeneration(
        #     nodes_path, edges_path, label, model, edgeType_control_data, True, ptm_name=ptm_name)
        draper_code = entry['tokenized']
        if graph_input_full is None:
            continue
        if label == 1:
            v += 1
        else:
            nv += 1
        if len(vuld_slices) > 0: vd_present += 1
        if len(syse_slices) > 0: syse_present += 1
        # if graph_input_control is not None: cg_present += 1
        # if graph_input_data is not None: dg_present += 1
        # if graph_input_cd is not None: cdg_present += 1
        data_point = {
            'id': didx,
            'file_name': file_name, 'file_path': os.path.abspath(entry['file_path']),
            'code': entry['code'],
            'vuld': vuld_slices, 'vd_present': 1 if len(vuld_slices) > 0 else 0,
            'syse': syse_slices, 'syse_present': 1 if len(syse_slices) > 0 else 0,
            'draper': draper_code,
            'full_graph': graph_input_full,
            # 'cgraph': graph_input_control,
            # 'dgraph': graph_input_data,
            # 'cdgraph': graph_input_cd,
            'label': int(entry['label'])
        }
        final_data.append(data_point)
        if len(final_data) == 5000:
            output_path = output_prefix + '.shard' + str(data_shard)
            with open(output_path, 'w') as fp:
                json.dump(final_data, fp)
                fp.close()
            print('Saved Shard %d to %s' % (data_shard, output_path), '=' * 100, 'Done', sep='\n')
            final_data = []
            data_shard += 1
    print("Vulnerable:\t%d\n"
          "Non-Vul:\t%d\n"
          "VulDeePecker:\t%d\n"
          "SySeVr:\t%d\n"
          "Control: %d\tData: %d\tBoth: %d" % \
          (v, nv, vd_present, syse_present, cg_present, dg_present, cdg_present))
    output_path = output_prefix + '.shard' + str(data_shard)
    with open(output_path, 'w') as fp:
        json.dump(final_data, fp)
        fp.close()
    print('Saved Shard %d to %s' % (data_shard, output_path), '=' * 100, 'Done', sep='\n')


def extract_graph_data(
    project, portion, base_dir='../data/full_experiment_real_data/', 
    output_dir='../data/full_experiment_real_data_processed/'):
    assert portion in ['full_graph', 'cgraph', 'dgraph', 'cdgraph']
    shards = os.listdir(base_dir)
    shard_count = len(shards)
    total_functions, in_scope_function = set(), set()
    vnt, nvnt = 0, 0
    graphs = []
    for sc in range(1, shard_count + 1):
        shard_file = open(os.path.join(base_dir, 'tmp_data.json.shard' + str(sc)))
        shard_data = json.load(shard_file)
        for data in tqdm(shard_data):
            fidx = data['id']
            label = int(data['label'])
            total_functions.add(fidx)
            present = data[portion] is not None
            code_graph = data[portion]
            if present:
                code_graph['id'] = fidx
                code_graph['file_name'] = data['file_name']
                code_graph['file_path'] = data['file_path']
                code_graph['code'] = data['code']
                graphs.append(code_graph)
                in_scope_function.add(fidx)
            else:
                if label == 1:
                    vnt += 1
                else:
                    nvnt += 1
        shard_file.close()
        del shard_data
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # tmp_mask = [0] * len(build_dataset_mask)
    # for item in graphs:
    #     tmp_mask[name2idx[item['file_name']]] = 1
    # align_mask(tmp_mask)
    output_file = open(os.path.join(output_dir, project + '-' + portion + '.json'), 'w')
    json.dump(graphs, output_file)
    output_file.close()
    print(project, portion, len(total_functions), len(in_scope_function), vnt, nvnt, sep='\t')

def full_data_prepare():
    base_dir = './tmp_dir'
    extract_graph_data('tmp', 'full_graph', base_dir=base_dir, output_dir='./')

def joern_parse(idx_list: list, code_list: list[str], label_list: list):
    cli = './code-slicer/joern/joern-parse'
    if os.path.exists('tmp_raw_code'):
        shutil.rmtree('tmp_raw_code')
    os.mkdir('tmp_raw_code')
    if os.path.exists('parsed'):
        shutil.rmtree('parsed')
    idx = 0
    for ids, rawcode, label in zip(idx_list, code_list, label_list):
        name = str(ids) + '_' + str(uuid.uuid4())[:8] + '_' + str(label) + '.c'
        if str(ids) == '183122':
            continue
        name2idx[name] = idx
        idx += 1
        with open(f'tmp_raw_code/{name}', 'w') as fp:
            fp.write(rawcode)
    result = subprocess.run([cli, os.path.abspath('tmp_raw_code')], capture_output=True, text=True)
    if result.returncode == 0:
        if os.path.exists('parsed'):
            cur_path = 'parsed'
            while not (cur_path.endswith('.c') or cur_path.endswith('.csv')) and len(os.listdir(cur_path)) != 0:
                # print(cur_path)
                cur_path = os.path.join(cur_path, os.listdir(cur_path)[0])
            if cur_path.endswith('.c') or cur_path.endswith('.csv'):
                proj_path = os.path.dirname(cur_path)
                if proj_path.endswith('.c'):
                    proj_path = os.path.dirname(proj_path)
                print(proj_path)
                # tmp_mask = [0] * len(build_dataset_mask)
                # for folder in os.listdir(proj_path):
                #     if folder.endswith('.c'):
                #         tmp_mask[name2idx[folder]] = 1
                # align_mask(tmp_mask)
                return proj_path
    print(result.stderr)
    print(result.stdout)
    return 'err'


# to align the source and input, we need to construct a mask list
build_dataset_mask = []
name2idx = {}

def align_mask(tgt_mask):
    assert len(tgt_mask) == len(build_dataset_mask)
    for idx, mask in enumerate(tgt_mask):
        if mask == 0:
            build_dataset_mask[idx] = 0


def build_model_input(idx_list: list, code_list: list[str], label_list: list, encode_type='wv', name=None):
    build_dataset_mask.clear()
    name2idx.clear()
    for idx, code in enumerate(code_list):
        build_dataset_mask.append(1)
    ret = joern_parse(idx_list, code_list, label_list)
    if ret == 'err':
        print('[ERROR]: joern parse failed')
        return
    # wv_path = './wv_models/li_et_al_wv'
    if name:
        if '-' in name:
            name = name.split('-')[0]
        wv_path = os.path.join('./wv_models', f"w2v_model_{name}.model")
    ptm_name = 'microsoft/codebert-base'
    if encode_type == 'wv':
        ptm_name = None
    elif encode_type == 'ptm':
        wv_path = None
    assert ptm_name == None or wv_path == None
    func_extract_slices('tmp_raw_code/', ret+'/')
    func_create_ggnn_data('tmp_raw_code/', ret+'/', wv_path=wv_path, ptm_name=ptm_name)
    full_data_prepare()
    if not os.path.exists('tmp-full_graph.json'):
        return
    return 'tmp-full_graph.json'


def jsonl_to_input(name, portion, path, encode_type, limit=5000):
    with open(path, 'r') as fp:
        samples = [json.loads(line) for line in fp.readlines()]
    idx_list = []
    code_list = []
    label_list = []
    for sam in samples:
        idx_list.append(sam['idx'])
        code_list.append(sam['func'])
        label_list.append(sam['target'])
    assert len(idx_list) == len(code_list) == len(label_list)
    if len(idx_list) > limit:
        fold = len(idx_list) // 5000 + 1
        idx_lists, code_lists, label_lists = [], [], []
        for i in range(fold):
            idx_lists.append(idx_list[i*limit:(i+1)*limit])
            code_lists.append(code_list[i*limit:(i+1)*limit])
            label_lists.append(label_list[i*limit:(i+1)*limit])
        print(f'Process over: {len(idx_lists)} times')
        ids = 0
        for idxl, codel, labell in zip(idx_lists, code_lists, label_lists):
            if build_model_input(idxl, codel, labell, encode_type, name=name) == 'tmp-full_graph.json':
                if not os.path.exists(os.path.join('../processed_data', f'{name}_full_data_processed')):
                    os.mkdir(os.path.join('../processed_data', f'{name}_full_data_processed'))
                shutil.move('./tmp-full_graph.json', os.path.join('../processed_data', f'{name}_full_data_processed', f'{name}_{portion}_group{ids}-full_graph.json'))
            ids += 1
    else:
        print('Process over: 1 times')
        if build_model_input(idx_list, code_list, label_list, encode_type, name=name) == 'tmp-full_graph.json':
            if not os.path.exists(os.path.join('../processed_data', f'{name}_full_data_processed')):
                os.mkdir(os.path.join('../processed_data', f'{name}_full_data_processed'))
            shutil.move('./tmp-full_graph.json', os.path.join('../processed_data', f'{name}_full_data_processed', f'{name}_{portion}-full_graph.json'))


def generate_w2vModel(
        data_paths: list[str],
        w2v_model_path: str,
        vector_dim=100,
        epochs=5,
        alpha=0.001,
        window=5,
        min_count=1,
        min_alpha=0.0001,
        sg=0,
        hs=0,
        negative=10
):
    print("Training w2v model...")
    sentences = []
    for path in data_paths:
        with open(path, 'r') as fp:
            samples = [json.loads(line) for line in fp.readlines()]
        for sam in samples:
            sentences.append(nltk.word_tokenize(sam['func']))
    print(len(sentences))
    model = Word2Vec(sentences=sentences, vector_size=vector_dim, alpha=alpha, window=window, min_count=min_count,
                     max_vocab_size=None, sample=0.001, seed=1, workers=8, min_alpha=min_alpha, sg=sg, hs=hs,
                     negative=negative)
    print('Embedding Size : ', model.vector_size)
    for _ in range(epochs):
        model.train(sentences, total_examples=len(sentences), epochs=1)
    model.save(w2v_model_path)
    return model


def train_wv_model(name, vector_dim=100):
    corpus_list = [
        '../dataset/VulFixed/train.jsonl',
        '../dataset/VulFixed/valid.jsonl',
        '../dataset/VulFixed/test.jsonl'
    ]
    wv_path = os.path.join('./wv_models', f"w2v_model_{name}.model")
    generate_w2vModel(corpus_list, wv_path, vector_dim=vector_dim)


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # ptm_model = AutoModel.from_pretrained("microsoft/codebert-base")
    # train_wv_model('Devign', vector_dim=100)
    jsonl_to_input('VulFixed', 'train', '../dataset/VulFixed/test.jsonl', 'wv')
    # main()
