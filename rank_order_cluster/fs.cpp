#include "fs.hpp"

#ifdef WIN32
#include <windows.h>
#include <io.h>
#include <direct.h>
// undef for fs api
#undef DeleteFile
#undef MoveFile
#else
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // WIN32

using std::vector;
using std::string;

namespace common {

namespace fs {

bool IsExists(const string& path) {
  if (access(path.c_str(), 0) != -1) {
    return true;
  }
  else {
    return false;
  }
}

bool MakeDir(const string& dpath) {
  if (IsExists(dpath) == true) {
    return true;
  }
  else {
    int rc;
#ifdef WIN32
    rc = mkdir(dpath.c_str());
#else
    // default mode, -rwxrwxr-x
    rc = mkdir(dpath.c_str(), 0775);
#endif  // WIN32
    if (rc == 0) {
      return true;
    }
    else {
      return false;
    }
  }
}

bool DeleteFile(const string& path) {
  if (remove(path.c_str()) == 0) {
    return true;
  }
  else {
    return false;
  }
}

bool MoveFile(const string& src_path, const string& dst_path) {
  if (rename(src_path.c_str(), dst_path.c_str()) == 0) {
    return true;
  }
  else {
    return false;
  }
}

string GetFileName(const string& path) {
  int pos = path.find_last_of("/");
  if (pos == string::npos) {  // xxx.yyy
    return path.substr(0, path.find_last_of("."));
  }
  else {  // /path/to/xxx.yyy
    string tmp = path.substr(pos + 1);
    return tmp.substr(0, tmp.find_last_of("."));
  }
}

string GetFileExtension(const string& path) {
  int pos = path.find_last_of(".");
  if (pos == string::npos) {
    return "";
  }
  else {
    return path.substr(pos + 1);
  }
}

vector<string> ListDir(const string& dpath, const vector<string>& exts) {
  vector<string> lists;
  if (!IsExists(dpath)) {
    return lists;
  }
#ifdef WIN32
  long fd;
  _finddata_t fi;
  for (int i = 0; i < exts.size(); i++) {
    string mp = dpath + "/*." + exts[i];
    if ((fd = _findfirst(mp.c_str(), &fi)) == -1) {  // no file
      continue;
    }
    do {
      lists.push_back(fi.name);
    } while (_findnext(fd, &fi) == 0);
    _findclose(fd);
  }
#else
  DIR *dir;
  struct dirent *file;
  struct stat st;
  if (!(dir = opendir(dpath.c_str()))) {
    return lists;
  }
  while ((file = readdir(dir)) != NULL) {
    string fn(file->d_name);
    // remove '.', '..'
    if (fn == "." || fn == "..") {
      continue;
    }
    string ext = GetFileExtension(fn);
    for (int i = 0; i < exts.size(); i++) {
      if (ext == exts[i]) {
        lists.push_back(fn);
        break;
      }
    }
  }
  closedir(dir);
#endif  // WIN32
  return lists;
}

vector<string> ListSubDir(const string& dpath) {
  vector<string> lists;
  if (!IsExists(dpath)) {
    return lists;
  }
#ifdef WIN32
  long fd;
  _finddata_t fi;
  string mp = dpath + "/*";
  if ((fd = _findfirst(mp.c_str(), &fi)) == -1) {  // no file
    return lists;
  }
  do {
    if (fi.attrib & _A_SUBDIR) {
      string dn(fi.name);
      if (dn != "." && dn != "..") {
        lists.push_back(fi.name);
      }
    }
  } while (_findnext(fd, &fi) == 0);
  _findclose(fd);
#else
  DIR *dir;
  struct dirent *file;
  struct stat st;
  if (!(dir = opendir(dpath.c_str()))) {
    return lists;
  }
  while ((file = readdir(dir)) != NULL) {
    string fn(file->d_name);
    string fp = dpath + "/" + fn;
    if (stat(fp.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
      // remove '.', '..'
      if (fn == "." || fn == "..") {
        continue;
      }
      lists.push_back(fn);
    }
  }
  closedir(dir);
#endif  // WIN32
  return lists;
}

}  // namespace fs

}  // namespace common
