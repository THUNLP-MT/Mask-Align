# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

CLI_DIR=$(cd "$(dirname "$0")"; pwd)

code_dir="$CLI_DIR/.."
export PYTHONPATH="$PYTHON:$code_dir"

EXP="DEFAULT"
suffix=""

SCRIPT=`basename ${BASH_SOURCE[0]}`

#Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}usage:${NORM} ${BOLD}bash $SCRIPT [-s CONFIG] [-e EXP] [-g] [-v] [-t] [-b] [-h]${NORM}"\\n
  echo "${REV}-s${NORM}  --provide config file. This argument is required."
  echo "${REV}-e${NORM}  --choose which experiment to run. Default is ${BOLD}DEFAULT${NORM}."
  echo "${REV}-g${NORM}  --set to generate attention weights."
  echo "${REV}-v${NORM}  --set to generate visualization data."
  echo "${REV}-t${NORM}  --set to test AER for alignment results."
  echo -e "${REV}-h${NORM}  --Displays this help message."\\n
  echo -e "Example: ${BOLD}bash $SCRIPT -s example.config -gtv"\\n
  exit 1
}

while getopts "s:e:gvtbh" OPT;
do
    case $OPT in
        s)
            CONFIG="$OPTARG";;
        e)
            EXP="$OPTARG";;
        g)
            suffix="$suffix --gen-weights";;
        v)
            suffix="$suffix --gen-vizdata";;
        t)
            suffix="$suffix --test-aer";;
        h)
            HELP;;
        ?)
            echo -e \\n"Option -${BOLD}${OPTARG}${NORM} not allowed."
            echo -e "Use ${BOLD}$SCRIPT -h${NORM} to see the help documentation."\\n;;
    esac
done

if [ "x" = "x$CONFIG" ]; then
    echo -e \\n"error: the following arguments are required: [-s CONFIG]"
    HELP
fi

echo "running $CONFIG"
python $CLI_DIR/inferrer.py --config $CONFIG --exp $EXP $suffix