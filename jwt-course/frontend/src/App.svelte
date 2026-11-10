<script>
  import { onMount } from 'svelte';
  import { UserPlus, Edit2, Trash2, LogOut, Users } from 'lucide-svelte';

  const API_URL = 'http://localhost:5317/api';

  let token = '';
  let username = '';
  let password = '';
  let participants = [];
  let loading = false;
  let error = '';
  let editingId = null;
  let formData = { name: '', email: '', phone: '' };

  onMount(() => {
    token = localStorage.getItem('token') || '';
    if (token) {
      fetchParticipants();
    }
  });

  async function login() {
    error = '';
    loading = true;

    try {
      const response = await fetch(`${API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await response.json();

      if (response.ok) {
        token = data.token;
        localStorage.setItem('token', data.token);
        username = '';
        password = '';
        await fetchParticipants();
      } else {
        error = data.message || 'Login gagal';
      }
    } catch (err) {
      error = 'Tidak dapat terhubung ke server';
    } finally {
      loading = false;
    }
  }

  function logout() {
    token = '';
    localStorage.removeItem('token');
    participants = [];
  }

  async function fetchParticipants() {
    loading = true;
    error = '';

    try {
      const response = await fetch(`${API_URL}/participants`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        participants = await response.json();
      } else {
        const data = await response.json();
        error = data.message || 'Gagal memuat data';
      }
    } catch (err) {
      error = 'Tidak dapat terhubung ke server';
    } finally {
      loading = false;
    }
  }

  async function addParticipant() {
    error = '';

    try {
      const response = await fetch(`${API_URL}/participants`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        formData = { name: '', email: '', phone: '' };
        await fetchParticipants();
      } else {
        const data = await response.json();
        error = data.message || 'Gagal menambah participant';
      }
    } catch (err) {
      error = 'Tidak dapat terhubung ke server';
    }
  }

  async function updateParticipant() {
    error = '';

    try {
      const response = await fetch(`${API_URL}/participants/${editingId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        formData = { name: '', email: '', phone: '' };
        editingId = null;
        await fetchParticipants();
      } else {
        const data = await response.json();
        error = data.message || 'Gagal mengupdate participant';
      }
    } catch (err) {
      error = 'Tidak dapat terhubung ke server';
    }
  }

  async function deleteParticipant(id) {
    if (!confirm('Yakin ingin menghapus participant ini?')) return;

    try {
      const response = await fetch(`${API_URL}/participants/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        await fetchParticipants();
      } else {
        const data = await response.json();
        error = data.message || 'Gagal menghapus participant';
      }
    } catch (err) {
      error = 'Tidak dapat terhubung ke server';
    }
  }

  function startEdit(participant) {
    editingId = participant.id;
    formData = {
      name: participant.name,
      email: participant.email,
      phone: participant.phone
    };
  }

  function cancelEdit() {
    editingId = null;
    formData = { name: '', email: '', phone: '' };
  }

  function handleSubmit() {
    if (editingId) {
      updateParticipant();
    } else {
      addParticipant();
    }
  }

  function handleKeyPress(e) {
    if (e.key === 'Enter') {
      login();
    }
  }
</script>

{#if !token}
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
      <div class="text-center mb-8">
        <div class="inline-block p-3 bg-indigo-100 rounded-full mb-4">
          <Users class="w-8 h-8 text-indigo-600" />
        </div>
        <h1 class="text-3xl font-bold text-gray-800">Login</h1>
        <p class="text-gray-600 mt-2">Participant Manager System</p>
      </div>

      <div class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Username
          </label>
          <input
            type="text"
            bind:value={username}
            on:keypress={handleKeyPress}
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            placeholder="admin"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Password
          </label>
          <input
            type="password"
            bind:value={password}
            on:keypress={handleKeyPress}
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            placeholder="password"
          />
        </div>

        {#if error}
          <div class="bg-red-50 text-red-700 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        {/if}

        <button
          on:click={login}
          disabled={loading}
          class="w-full bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 transition disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Login'}
        </button>

        <p class="text-sm text-gray-500 text-center mt-4">
          Default: admin / password
        </p>
      </div>
    </div>
  </div>
{:else}
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
      <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
        <div class="bg-gradient-to-r from-indigo-600 to-purple-600 p-6 flex justify-between items-center">
          <div class="flex items-center gap-3">
            <Users class="w-8 h-8 text-white" />
            <h1 class="text-2xl font-bold text-white">Participant Manager</h1>
          </div>
          <button
            on:click={logout}
            class="flex items-center gap-2 bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg transition"
          >
            <LogOut class="w-4 h-4" />
            Logout
          </button>
        </div>

        <div class="p-6">
          {#if error}
            <div class="bg-red-50 text-red-700 px-4 py-3 rounded-lg mb-6">
              {error}
            </div>
          {/if}

          <div class="mb-8 bg-gray-50 p-6 rounded-xl">
            <h2 class="text-xl font-semibold mb-4 text-gray-800">
              {editingId ? 'Edit Participant' : 'Tambah Participant Baru'}
            </h2>
            
            <div class="grid md:grid-cols-3 gap-4 mb-4">
              <input
                type="text"
                placeholder="Nama"
                bind:value={formData.name}
                class="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
              <input
                type="email"
                placeholder="Email"
                bind:value={formData.email}
                class="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
              <input
                type="tel"
                placeholder="No. Telepon"
                bind:value={formData.phone}
                class="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div class="flex gap-2">
              <button
                on:click={handleSubmit}
                class="flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition"
              >
                <UserPlus class="w-4 h-4" />
                {editingId ? 'Update' : 'Tambah'}
              </button>
              {#if editingId}
                <button
                  on:click={cancelEdit}
                  class="bg-gray-300 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-400 transition"
                >
                  Batal
                </button>
              {/if}
            </div>
          </div>

          <div>
            <h2 class="text-xl font-semibold mb-4 text-gray-800">
              Daftar Participants ({participants.length})
            </h2>

            {#if loading && participants.length === 0}
              <div class="text-center py-8 text-gray-500">Loading...</div>
            {:else if participants.length === 0}
              <div class="text-center py-8 text-gray-500">Belum ada participant</div>
            {:else}
              <div class="space-y-3">
                {#each participants as participant (participant.id)}
                  <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                    <div class="flex justify-between items-center">
                      <div class="flex-1">
                        <h3 class="font-semibold text-gray-800">{participant.name}</h3>
                        <div class="text-sm text-gray-600 mt-1">
                          <span>{participant.email}</span>
                          <span class="mx-2">•</span>
                          <span>{participant.phone}</span>
                        </div>
                      </div>
                      <div class="flex gap-2">
                        <button
                          on:click={() => startEdit(participant)}
                          class="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition"
                        >
                          <Edit2 class="w-5 h-5" />
                        </button>
                        <button
                          on:click={() => deleteParticipant(participant.id)}
                          class="p-2 text-red-600 hover:bg-red-50 rounded-lg transition"
                        >
                          <Trash2 class="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }
</style>