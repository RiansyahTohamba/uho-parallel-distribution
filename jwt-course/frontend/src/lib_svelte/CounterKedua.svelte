<script>
  let kounter = $state(20)
  const tambahNilai = () => {
    kounter += 1
  }
</script>
<p>Counter ke 2</p>
<button onclick={tambahNilai}>
  Nilai variable count adalah {kounter}
</button>
