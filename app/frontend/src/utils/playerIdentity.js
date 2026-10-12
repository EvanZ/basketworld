export const PLAYER_SURNAME_POOL = [
  'Smith', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor',
  'Thomas', 'White', 'Clark', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen',
  'Young', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Green',
  'Adams', 'Baker', 'Nelson', 'Carter', 'Perez', 'Turner', 'Parker', 'Evans',
  'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey',
  'Rivera', 'Cooper', 'Cox', 'Gray', 'James', 'Watson', 'Brooks', 'Kelly',
  'Price', 'Wood', 'Barnes', 'Ross', 'Cole', 'Perry', 'Powell', 'Long',
  'Patel', 'Flores', 'Hughes', 'Butler', 'Foster', 'Bryant', 'Diaz', 'Hayes',
  'Myers', 'Ford', 'Graham', 'Woods', 'West', 'Jordan', 'Owens', 'Fisher',
  'Ellis', 'Harris', 'Hudson', 'Ryan', 'Porter', 'Hunter', 'Hicks', 'Henry',
  'Bishop', 'Dean', 'Mason', 'Hart', 'Willis', 'Lane', 'Riley', 'Rose',
  'Stone', 'Dunn', 'Payne', 'Ray', 'Berry', 'Arnold', 'Wagner', 'Weaver',
  'Burke', 'Lynch', 'Hanson', 'Day', 'Fox', 'Ramsey', 'Shaw', 'Burns',
  'Gordon', 'Warren', 'Dixon', 'Ramos', 'Reyes', 'Cruz', 'Ortiz', 'Soto',
  'Kim', 'Li', 'Ho', 'Tran', 'Pham', 'Le',
];

export function playerSurnameForId(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id) || PLAYER_SURNAME_POOL.length === 0) return '';
  const index = ((Math.trunc(id) % PLAYER_SURNAME_POOL.length) + PLAYER_SURNAME_POOL.length)
    % PLAYER_SURNAME_POOL.length;
  return String(PLAYER_SURNAME_POOL[index] || '').toUpperCase();
}
