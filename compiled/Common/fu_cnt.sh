for i in "Max" "Add" "Mul" "Sub" "Div" "Sqrt" "Compare" "Input" "Output"
do
  echo $i `cat $1 | grep "$i" | wc -l`
done
