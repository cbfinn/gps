import re
import os.path as osp

fname = "../mjc2/Include/mj_engine_typedef.h"
with open(fname,"r") as fh:
    all_lines = fh.readlines()

def find_lines_between(lines, start_substring, end_substring):
    it = iter(lines)
    for line in it:
        if start_substring in line:
            break
    outlines = []
    for line in it:
        if end_substring in line: break
        else: outlines.append(line)
    return outlines



scalar_re = re.compile("\s+(\w+)\s+(\w+);")
arrsize=".+\(([\w\+\*]+) x (\w+)\)"
ptr_re = re.compile("\s+(\w+)\*\s+(\w+);"+arrsize)
arr_re = re.compile("\s+(\w+)\s(\w+)\[(\w+)\];")

def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def isvalidsize(s):
    return (s in int_names) or isint(s)
def ismacro(s):
    return s.lower() != s and s != "nM"
def add_model_to_fields(s):
    md = re.match("^\w+$",s)
    if md: 
        if "*" in s: asdf
        if ismacro(s) or isint(s):
            return s
        else:
            return "m_model->%s"%s
    if s == "nlmax+ncmax":
        return "m_model->nlmax+m_model->ncmax"
    elif s == "nwrap*2":
        return "m_model->nwrap*2"
    else:
        raise Exception

    

def process(in_lines,structname):
    get_lines = []
    set_lines = []

    for line in in_lines:
        md = scalar_re.match(line)
        if md:
            dtype,name = md.group(1),md.group(2)
            get_lines.append('    out["%(name)s"] = %(structname)s->%(name)s;'%dict(structname=structname,dtype=dtype,name=name))            
            set_lines.append('    _csdihk(d, "%(name)s", %(structname)s->%(name)s);'%dict(structname=structname,name=name))            
            continue
        md = ptr_re.match(line)
        if md:
            dtype,name,size0,size1 = md.group(1),md.group(2),md.group(3),md.group(4)
            if dtype == "mjContact":
                print "skipping due to dtype",line
                continue
            size0 = add_model_to_fields(size0)
            size1 = add_model_to_fields(size1)
            get_lines.append('    out["%(name)s"] = toNdarray2<%(dtype)s>(%(structname)s->%(name)s, %(size0)s, %(size1)s);'%dict(structname=structname,name=name,dtype=dtype,size0=size0,size1=size1))            
            set_lines.append('    _cadihk(d, "%(name)s", %(structname)s->%(name)s);'%dict(structname=structname,name=name))            
            continue
        md = arr_re.match(line)
        if md:
            dtype,name,size = md.group(1),md.group(2),md.group(3)
            get_lines.append('    out["%(name)s"] = toNdarray1<%(dtype)s>(%(structname)s->%(name)s, %(size)s);'%dict(structname=structname,name=name,dtype=dtype,size=size))            
            set_lines.append('    _cadihk(d, "%(name)s", %(structname)s->%(name)s);'%dict(structname=structname,name=name))            
            continue
        print "ignore line:",line,
    return get_lines,set_lines



with open("mjcpy_getmodel_autogen.i","w") as outfile:
    get_lines, set_lines = process(find_lines_between(all_lines, "_mjModel","}"),"m_model")
    outfile.write("\n".join(get_lines))
with open("mjcpy_getoption_autogen.i","w") as outfile:
    get_lines, set_lines = process(find_lines_between(all_lines, "_mjOption","}"),"m_option")
    outfile.write("\n".join(get_lines))
with open("mjcpy_setoption_autogen.i", "w") as outfile:
    get_lines, set_lines = process(find_lines_between(all_lines, "_mjOption","}"),"m_option")
    outfile.write("\n".join(set_lines))
with open("mjcpy_getdata_autogen.i","w") as outfile:
    get_lines, set_lines = process(find_lines_between(all_lines, "_mjData","}"),"m_data")
    outfile.write("\n".join(get_lines))

