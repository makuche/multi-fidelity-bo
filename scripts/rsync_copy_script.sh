#!/bin/sh

# Simple rsync "driver" script.  (Uses SSH as the transport layer.)
# http://www.scrounge.org/linux/rsync.html

# Demonstrates how to use rsync to back up a directory tree from a local
# machine to a remote machine.  Then re-run the script, as needed, to keep
# the two machines "in sync."  It only copies new or changed files and ignores
# identical files.

# NOTE: the script can copy between very different locations,
# just be careful how you define DEST and what is local.

################# Define locations #################

# Destination host machine name: can be local or remote
#DEST="mycomputer.mydomain"
#DEST="150.244.203.109"
DEST=puhti.csc.fi

# User that rsync will connect as
#USER="root"
USER=myusername

# Directory to copy from on the source machine.
# End with / to copy dir. contents. Omit to copy entire dir.
#DESTDIR="/wrk/mtodorov/"
DESTDIR="/scratch/somedir/user/"

# Directory to copy to on the destination machine.
#BACKDIR="/home/user/BACKUPS/"
BACKDIR="/l/user/CSC_BACKUP/"

# excludes file - Contains wildcard patterns of files to exclude.
# i.e., *~, *.bak, etc.  One "pattern" per line.
# You must create this file if you wish to use this.
# EXCLUDES=/root/bin/excludes

######################  Options #########################
# -n Don't do any copying, but display what rsync *would* copy. For testing.
# -a Archive. Mainly propogate file permissions, ownership, timestamp, etc.
# -u Update. Don't copy file if file on destination is newer.
# -v Verbose -vv More verbose. -vvv Even more verbose.
# See man rsync for other options.

# For testing.  Only displays what rsync *would* do and does no actual copying.
#OPTS="-n -vv -u -a --rsh=ssh --exclude-from=$EXCLUDES --stats --progress"
#OPTS="-n -vv -razvulpt -e ssh --stats --progress"

# Does copy, but still gives a verbose display of what it is doing
OPTS="-razvulpt -e ssh --stats --progress --timeout=3600"
#OPTS="-v -u -a --rsh=ssh --exclude-from=$EXCLUDES --stats"

# Copies and does no display at all.
#OPTS="--archive --update --rsh=ssh --exclude-from=$EXCLUDES --quiet"

################ Finally: rsync ###################
rsync $OPTS $USER@$DEST:$DESTDIR $BACKDIR > rsync.log
