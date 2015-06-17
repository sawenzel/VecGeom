#!/usr/bin/python
# Copyright (C) 2010-2011 BUGSENG srl (http://bugseng.com)
# All rights reserved worldwide.

import os.path
import re
import fnmatch
import sqlite3

def outheader(out, dir, count):
   if count == 1:
      out.write('<tr bgcolor="#FFFFFF"><td><b><a name="{0}"></a>Directory {0}: {1} violation</b></td></tr>\n'.format(dir, count))
   else:
      out.write('<tr bgcolor="#FFFFFF"><td><b><a name="{0}"></a>Directory {0}: {1} violations</b></td></tr>\n'.format(dir, count))

def outincludes(out, info, cnt):
   # write includes
   dir = ''
   nrow = 0
   info.sort()
   for inc in info:
      r = os.path.dirname(inc[0])
      if dir != r:
         dir = r
         outheader(out, r, cnt[r])
         nrow = 0
      nrow += 1
      outline(out, inc, nrow)

def outline(out, info, nrow):
   if nrow % 2:
      bgc = 'bgcolor="#CCCCCC"'
   else:
      bgc = 'bgcolor="#DDDDDD"'
   out.write('\n')
   outp = str(info[0])
   output = outp.split('/')
   if (output[0] == 'VecGeom'):
	del output[0]
	file_name = "/".join(output)
  	out.write('<tr {4}><td><a href="http://git.cern.ch/pubweb/VecGeom.git/blob/HEAD:{0}#l{1}">{0}:{1}</a>:{2}: {3}</td></tr>'.format(file_name, info[1], info[2], info[3], bgc))

def report(file):
    db = sqlite3.connect(file)

    q1 = db.cursor()
    q1.execute('select service, dirname, count(*) from hreport cross join area, hfile where area.hreport = hreport.id and area.variant = "" and area.idx = 1 and hfile.id = area.hfile1 group by service, dirname order by service, dirname')

    nrow = 0
    old_rule = ''
    includes = { }
    inclcnt = { }
    inclinfo = []
    for row1 in q1:

        (rule, dir, count) = row1

        if old_rule != rule:
            if old_rule != '':
                # write includes
                outincludes(out, inclinfo, inclcnt)
                out.write('\n</tbody></table>\n')
                out.write('</body></html>\n')
                out.close()
                nrow = 0
                includes = { }
                inclcnt = { }
                inclinfo = []
            old_rule = rule
            print 'Processing violations of rule ' + rule + '...'
            out = open(rule + '.html', 'w')
            q2 = db.cursor()
            q2.execute('select summary from service where id = ?', (rule,))
            (summary,) = q2.fetchone()
            out.write('<html><head> </head><body>\n')
            out.write('<h3>Rule {0}: {1}</h3>\n'.format(rule, summary))
            out.write('<table CELLSPACING=0 CELLPADDING=4 BGCOLOR="#FFFFFF" border="0" cols="1" width="100%"><tbody>\n')

        q3 = db.cursor()
        q3.execute('select pathname, line1, column1, message from hreport cross join area, hfile where service = ? and dirname = ? and area.hreport = hreport.id and area.variant = "" and area.idx = 1 and hfile.id = area.hfile1 order by pathname', (rule, dir))

        if dir.find('include', 0, 7) == -1:
           outheader(out, dir, count)
           nrow = 0

        for row3 in q3:
            (f,l,c,mess) = row3
            isinc = 0
            if f.find('include/', 0, 8) == 0:
               isinc = 1
               inc = f.replace('include/', '')
               if inc in includes:
                  f = includes[inc]
                  inclinfo.append((f, l, c, mess))
                  r = os.path.dirname(f)
                  inclcnt[r] += 1
               else:
                  for root, dirnames, filenames in os.walk('.'):
                     for filename in fnmatch.filter(filenames, inc):
                        if root.find('./include', 0, 9) == -1:
                           r = root.replace('./', '')
                           f = os.path.join(r, filename)
                           includes[inc] = f
                           inclinfo.append((f, l, c, mess))
                           if r in inclcnt:
                              inclcnt[r] += 1
                           else:
                              inclcnt[r] = 1

            if isinc == 0:
               nrow += 1
               outline(out, (f, l, c, mess), nrow)

    # write includes
    outincludes(out, inclinfo, inclcnt)

    if rule != '':
        out.write('\n</tbody></table>\n')
        out.write('</body></html>\n')
        out.close()

import sys
import getopt

def usage():
    print
    sys.exit(2)

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.GetoptError, msg:
        print str(msg)
        usage()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
    if len(args) != 1:
        usage()
    file = args[0]
    report(file)

if __name__ == '__main__':
    main()
