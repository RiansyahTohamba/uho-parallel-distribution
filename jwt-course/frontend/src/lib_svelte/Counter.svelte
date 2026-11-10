<script>
  let count = $state(5)
  const tambahNilai = () => {
    count += 1
  }
</script>
<p>Counter pertama</p>
<button onclick={tambahNilai}>
  Nilai variable count adalah {count}
</button>
